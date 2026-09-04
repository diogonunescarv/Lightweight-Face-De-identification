import math
import time
from pathlib import Path
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.transformer import LightweightDeIdentifier
from models.embedders import load_authorized_embedders
from models.masks import (
    detect_landmarks_batch,
    parse_mask_regions,
    parse_mask_shape,
    parse_ssim_region,
    resolve_face_mask,
    resolve_ssim_mask,
)
from data.dataset import FaceImageFolder, save_preview
from losses.losses import identity_loss_ensemble, ssim_index_masked, pixel_l2_loss, total_variation_loss
from evaluation.validate import (
    compute_validation_metrics,
    early_stopping_score,
    is_improvement,
)
from utils.scheduler import WarmupScheduler
from utils.helpers import set_seed


def train(args, *, val_callback=None):
    set_seed(args.seed)

    device = torch.device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "steps.csv"
    with open(log_path, "w") as f:
        f.write("step,loss,id,cos,ssim,pix,tv,elapsed,lr,euclid\n")

    val_log_path = None
    if args.early_stopping:
        val_log_path = out_dir / "val.csv"
        with open(val_log_path, "w") as f:
            f.write("step,cos_mean,euclid_mean,ssim_mean,score,is_best\n")

    dataset = FaceImageFolder(args.data, args.image_size)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )

    val_loader = None
    if args.early_stopping:
        val_dataset = FaceImageFolder(args.val_data, args.image_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        print(
            f"Early stopping: val={args.val_data} eval_every={args.eval_every} "
            f"patience={args.patience} metric={args.early_stopping_metric} "
            f"val_max_samples={args.val_max_samples}"
        )

    embedders = load_authorized_embedders(device)

    mask_regions = parse_mask_regions(args.mask_regions)
    mask_shape = parse_mask_shape(args.mask_shape)
    ssim_region = parse_ssim_region(args.ssim_region)
    if args.mask_mode == "fixed" and args.mask_regions != "full":
        print(
            f"Aviso: --mask-regions={args.mask_regions} ignorado com --mask-mode fixed "
            "(usa elipse centrada original)."
        )
    if args.mask_mode == "fixed" and mask_shape != "ellipse":
        print(
            f"Aviso: --mask-shape={mask_shape} ignorado com --mask-mode fixed "
            "(usa elipse centrada original)."
        )

    transformer = LightweightDeIdentifier(
        image_size=args.image_size,
        transform_type=args.transform_type,
        dct_k=args.dct_k,
        dct_fmin=args.dct_fmin,
        dct_fmax=args.dct_fmax,
        wavelet_J=args.wavelet_J,
        max_wavelet_amp=args.max_wavelet_amp,
        flow_grid=args.flow_grid,
        photo_grid=args.photo_grid,
        max_dct_amp=args.max_dct_amp,
        max_flow_px=args.max_flow_px,
        max_photo_amp=args.max_photo_amp,
        use_face_mask=not args.no_face_mask,
        mask_mode=args.mask_mode,
        mask_regions=mask_regions,
        mask_shape=mask_shape,
        disable_dct=args.disable_dct,
        disable_flow=args.disable_flow,
        disable_photo=args.disable_photo,
    ).to(device)

    landmark_detector = None
    need_landmarks = (
        not args.no_face_mask
        and (args.mask_mode == "landmarks" or ssim_region == "full-landmarks")
    )
    if need_landmarks:
        from models.face_detector import get_default_detector

        landmark_detector = get_default_detector()
        if args.mask_mode == "landmarks":
            print(
                f"Máscaras por landmarks: shape={mask_shape} "
                f"regions={','.join(mask_regions)}"
            )
    print(f"SSIM region: {ssim_region}")

    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)

    scheduler = None
    warmup_steps = args.lr_warmup_steps

    if args.lr_scheduler == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps - warmup_steps, eta_min=args.lr_eta_min
        )
        print(f"Usando CosineAnnealingLR: T_max={args.steps - warmup_steps}, eta_min={args.lr_eta_min}")
        scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_steps, args.lr)
    elif args.lr_scheduler == "step":
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
        print(f"Usando StepLR: step_size={args.lr_step_size}, gamma={args.lr_gamma}")
        scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_steps, args.lr)
    elif args.lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.steps,
            pct_start=args.onecycle_pct_start,
            anneal_strategy=args.onecycle_anneal_strategy
        )
        print(f"Usando OneCycleLR: max_lr={args.lr}, total_steps={args.steps}, "
              f"pct_start={args.onecycle_pct_start}, anneal_strategy={args.onecycle_anneal_strategy}")
    else:
        print("Usando LR fixo (sem scheduler)")

    step = 0
    start = time.time()
    early_stopped = False
    best_score = float("inf") if args.early_stopping_metric == "cos" else float("-inf")
    best_step = 0
    best_metrics = None
    patience_counter = 0
    num_evals = 0
    best_ckpt_path = out_dir / "transform_best.pt"

    while step < args.steps:
        for x, _paths in loader:
            if step >= args.steps:
                break

            x = x.to(device)

            landmarks_batch = None
            if need_landmarks:
                landmarks_batch = detect_landmarks_batch(_paths, landmark_detector)

            face_mask = resolve_face_mask(
                transformer,
                x.shape[0],
                _paths,
                device,
                use_face_mask=not args.no_face_mask,
                mask_mode=args.mask_mode,
                mask_regions=mask_regions,
                mask_shape=mask_shape,
                detector=landmark_detector,
                landmarks_batch=landmarks_batch,
            )
            y = transformer(x, face_mask=face_mask)

            loss_id, mean_cos = identity_loss_ensemble(
                embedders=embedders,
                x_original=x,
                x_transformed=y,
                target_cos=args.target_cos,
            )

            ssim_mask = resolve_ssim_mask(
                transformer,
                _paths,
                device,
                ssim_region=ssim_region,
                face_mask=face_mask,
                mask_mode=args.mask_mode,
                mask_regions=mask_regions,
                mask_shape=mask_shape,
                landmarks_batch=landmarks_batch,
                detector=landmark_detector,
            )
            ssim_vals = ssim_index_masked(x, y, ssim_mask)
            loss_ssim = F.relu(args.tau_ssim - ssim_vals).mean()

            loss_tv = total_variation_loss(y)

            regs = transformer.regularization()

            loss = (
                args.lambda_id * loss_id
                + args.lambda_ssim * loss_ssim
                + args.lambda_tv * loss_tv
                + args.lambda_flow_smooth * regs["flow_smooth"]
                + args.lambda_dct_l2 * regs.get("dct_l2", 0.0)
                + args.lambda_photo_l2 * regs["photo_l2"]
            )

            if args.transform_type == "dtcwt":
                loss += args.lambda_wavelet_mag * regs.get("wavelet_mag_l2", 0.0)
                loss += args.lambda_wavelet_phase * regs.get("wavelet_phase_l2", 0.0)
                loss += args.lambda_wavelet_smooth * regs.get("wavelet_mag_smooth", 0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                elapsed = time.time() - start

                current_lr = optimizer.param_groups[0]['lr']

                euclid = math.sqrt(2.0 * (1.0 - mean_cos.item()))

                with open(log_path, "a") as f:
                    f.write(
                        f"{step},"
                        f"{loss.item():.5f},"
                        f"{loss_id.item():.5f},"
                        f"{mean_cos.item():.4f},"
                        f"{ssim_vals.mean().item():.4f},"
                        f"{loss_tv.item():.6f},"
                        f"{elapsed:.1f},"
                        f"{current_lr:.6f},"
                        f"{euclid:.4f}\n"
                    )

                print(
                    f"[step {step:06d}] "
                    f"loss={loss.item():.5f} "
                    f"id={loss_id.item():.5f} "
                    f"cos={mean_cos.item():.4f} "
                    f"euclid={euclid:.4f} "
                    f"ssim={ssim_vals.mean().item():.4f} "
                    f"tv={loss_tv.item():.6f} "
                    f"elapsed={elapsed:.1f}s "
                    f"lr={current_lr:.6f}\n"
                )

            if step % args.preview_every == 0:
                save_preview(x, y, preview_dir / f"step_{step:06d}.jpg")

            if args.early_stopping and step > 0 and step % args.eval_every == 0:
                metrics = compute_validation_metrics(
                    transformer,
                    embedders,
                    val_loader,
                    device,
                    mask_mode=args.mask_mode,
                    mask_regions=mask_regions,
                    mask_shape=mask_shape,
                    ssim_region=ssim_region,
                    use_face_mask=not args.no_face_mask,
                    landmark_detector=landmark_detector,
                    max_samples=args.val_max_samples,
                )
                score = early_stopping_score(metrics, args.early_stopping_metric)
                num_evals += 1

                improved = is_improvement(
                    score,
                    best_score,
                    metric_name=args.early_stopping_metric,
                    min_delta=args.early_stopping_min_delta,
                )
                if improved:
                    best_score = score
                    best_step = step
                    best_metrics = dict(metrics)
                    patience_counter = 0
                    save_checkpoint(
                        transformer,
                        args,
                        best_ckpt_path,
                        step,
                        extra={
                            "early_stopping_score": score,
                            "early_stopping_metric": args.early_stopping_metric,
                        },
                    )
                else:
                    patience_counter += 1

                if val_log_path is not None:
                    with open(val_log_path, "a") as f:
                        f.write(
                            f"{step},"
                            f"{metrics['cos_mean']:.6f},"
                            f"{metrics['euclid_mean']:.6f},"
                            f"{metrics['ssim_mean']:.6f},"
                            f"{score:.6f},"
                            f"{1 if improved else 0}\n"
                        )

                print(
                    f"[val step {step:06d}] "
                    f"cos={metrics['cos_mean']:.4f} "
                    f"euclid={metrics['euclid_mean']:.4f} "
                    f"ssim={metrics['ssim_mean']:.4f} "
                    f"score={score:.4f} "
                    f"best={best_score:.4f}@{best_step} "
                    f"patience={patience_counter}/{args.patience}\n"
                )

                if val_callback is not None:
                    val_callback(step, score, metrics)

                if patience_counter >= args.patience:
                    early_stopped = True
                    print(
                        f"Early stopping at step {step} "
                        f"(best step {best_step}, best {args.early_stopping_metric}={best_score:.4f}, "
                        f"patience exhausted)."
                    )
                    break

                transformer.train()

            elif step % args.save_every == 0 and step > 0 and not args.early_stopping:
                save_checkpoint(transformer, args, out_dir / "transform.pt", step)

            step += 1

        if early_stopped:
            break

    extra_meta = {}
    if args.early_stopping and num_evals > 0 and best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        transformer.load_state_dict(best_ckpt["state_dict"])
        extra_meta = {
            "early_stopped": early_stopped,
            "best_step": best_step,
            "best_score": best_score,
            "stopped_step": step,
            "early_stopping_metric": args.early_stopping_metric,
        }
        save_checkpoint(transformer, args, out_dir / "transform.pt", best_step, extra=extra_meta)
        print(
            f"Modelo salvo em: {out_dir / 'transform.pt'} "
            f"(best step {best_step}, early_stopped={early_stopped})"
        )
    else:
        save_checkpoint(transformer, args, out_dir / "transform.pt", step, extra=extra_meta)
        print(f"Modelo salvo em: {out_dir / 'transform.pt'}")

    print(f"Log salvo em: {log_path}")
    if val_log_path is not None:
        print(f"Log de validação salvo em: {val_log_path}")

    result = {
        "best_score": best_score if num_evals > 0 else None,
        "best_step": best_step if num_evals > 0 else None,
        "stopped_step": step,
        "early_stopped": early_stopped,
        "cos_mean": best_metrics["cos_mean"] if best_metrics else None,
        "ssim_mean": best_metrics["ssim_mean"] if best_metrics else None,
        "euclid_mean": best_metrics["euclid_mean"] if best_metrics else None,
        "out_dir": str(out_dir),
    }
    return result


def save_checkpoint(
    transformer: LightweightDeIdentifier,
    args,
    path: Path,
    step: int,
    extra: dict | None = None,
):
    ckpt = {
        "step": step,
        "image_size": args.image_size,
        "dct_k": args.dct_k,
        "dct_fmin": args.dct_fmin,
        "dct_fmax": args.dct_fmax,
        "flow_grid": args.flow_grid,
        "photo_grid": args.photo_grid,
        "max_dct_amp": args.max_dct_amp,
        "max_flow_px": args.max_flow_px,
        "max_photo_amp": args.max_photo_amp,
        "use_face_mask": not args.no_face_mask,
        "mask_mode": args.mask_mode,
        "mask_regions": args.mask_regions,
        "mask_shape": args.mask_shape,
        "ssim_region": args.ssim_region,
        "transform_type": transformer.transform_type,
        "wavelet_J": transformer.wavelet_J,
        "max_wavelet_amp": transformer.max_wavelet_amp,
        "state_dict": transformer.state_dict(),
        "disable_dct": args.disable_dct,
        "disable_flow": args.disable_flow,
        "disable_photo": args.disable_photo,
    }
    if extra:
        ckpt.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
