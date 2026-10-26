"""V2 trainer: DT-CWT meta-mask + ROI pixel compose (Ideia 1) + Phase 3.3 loss."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from data.merged_split import (
    MERGED_DATASET,
    build_flat_data_dict,
    build_split_config_block,
    make_face_dataloader,
    observe_batch_hw,
    sample_native_hw,
)
from evaluation.validate_v2 import (
    early_stopping_score_v2,
    is_improvement_v2,
    validate_v2,
)
from losses.identity import identity_cosine_loss
from losses.quality import adapt_percep_weight, ssim_hinge_loss
from models.dtcwt_meta_mask import (
    DEFAULT_EPSILON,
    DTCWTMetaMaskModule,
    parse_editable_levels,
)
from models.fr import (
    SMOKE_FR_NAMES,
    load_fr_ensemble,
    resolve_eval_fr_names,
    resolve_train_fr_names,
    assert_train_eval_disjoint,
)
from models.masks import (
    detect_landmarks_batch,
    parse_mask_regions,
    parse_mask_shape,
    parse_ssim_region,
    resolve_face_mask_by_size,
    resolve_ssim_mask_by_size,
)
from utils.helpers import set_seed

CHECKPOINT_LATEST = "checkpoint_latest.pt"
META_MASK_BEST = "meta_mask_best.pt"
META_MASK_FINAL = "meta_mask.pt"


def protego_sign_step(module: torch.nn.Module, lr: float) -> None:
    """Protego / SignSGD: ``param -= lr * sign(grad)`` (no coeff L∞ projection)."""
    with torch.no_grad():
        for p in module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            p.add_(-lr * p.grad.sign())
            p.grad = None


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_preview(
    x: torch.Tensor,
    x_prot: torch.Tensor,
    face_mask: torch.Tensor,
    path: Path,
    gain: float = 8.0,
) -> None:
    from data.dataset import tensor_to_pil
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    i = 0
    o = tensor_to_pil(x[i])
    p = tensor_to_pil(x_prot[i])
    resid = (x_prot[i] - x[i]).abs().mean(dim=0).clamp(0, 1) * gain
    resid = resid.clamp(0, 1)
    r = Image.fromarray(
        (resid.detach().cpu().numpy() * 255).round().astype(np.uint8)
    ).convert("RGB")
    m = face_mask[i, 0].detach().cpu().numpy()
    overlay = np.asarray(o).astype(np.float32).copy()
    heat = (m * 255).clip(0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] * 0.5 + heat * 0.5, 0, 255)
    ov = Image.fromarray(overlay.astype(np.uint8))
    canvas = Image.new("RGB", (o.width * 4, o.height))
    for idx, im in enumerate((o, p, r, ov)):
        canvas.paste(im.resize((o.width, o.height)), (idx * o.width, 0))
    canvas.save(path)


def _default_lr(optimizer: str, epsilon: float, lr: Optional[float]) -> float:
    if lr is not None:
        return float(lr)
    if optimizer == "protego":
        return 0.01 * float(epsilon)
    return 1e-2


def train_v2(args) -> Path:
    """
    Run V2 training. All artifacts stay under ``args.out`` (or resume folder).

    Returns the run directory path.
    """
    set_seed(int(getattr(args, "seed", 42)))

    roi_compose = getattr(args, "roi_compose", "pixel")
    if roi_compose == "wavelet":
        raise NotImplementedError(
            "roi_compose='wavelet' disponível no Step 7. Use --roi-compose pixel."
        )
    if roi_compose != "pixel":
        raise ValueError(f"roi_compose inválido: {roi_compose!r}")

    device = torch.device(getattr(args, "device", "cpu"))
    image_size = int(getattr(args, "image_size", 224))
    epsilon = float(getattr(args, "epsilon", DEFAULT_EPSILON))
    optimizer_name = str(getattr(args, "optimizer", "protego")).lower()
    lr = _default_lr(optimizer_name, epsilon, getattr(args, "lr", None))
    min_ssim = float(getattr(args, "min_ssim", 0.95))
    epochs = int(getattr(args, "epochs", 5))
    batch_size = int(getattr(args, "batch_size", 2))
    init_std = float(getattr(args, "init_std", 1e-3))
    editable = parse_editable_levels(
        getattr(args, "editable_levels", "2,3"),
        int(getattr(args, "wavelet_J", 3)),
    )
    J = int(getattr(args, "wavelet_J", 3))

    use_face_mask = not bool(getattr(args, "no_face_mask", False))
    mask_mode = str(getattr(args, "mask_mode", "landmarks"))
    mask_shape = parse_mask_shape(getattr(args, "mask_shape", "ellipse"))
    mask_regions = parse_mask_regions(getattr(args, "mask_regions", "full"))
    ssim_region = parse_ssim_region(getattr(args, "ssim_region", "full-landmarks"))

    resume_path = getattr(args, "resume", None)
    if resume_path:
        out_dir = Path(resume_path).resolve().parent
    else:
        out_root = Path(getattr(args, "out_root", getattr(args, "out", "./runs")))
        exp_name = getattr(args, "exp_name", "v2_run")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"{stamp}_v2_{exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    # ---- data ----
    train_paths = getattr(args, "train_paths", None)
    eval_paths = getattr(args, "eval_paths", None)
    split_stats: dict[str, int] = {}
    if train_paths is None:
        data = build_flat_data_dict(
            MERGED_DATASET,
            celeba_train_frac=float(getattr(args, "celeba_train_frac", 0.5)),
            data_seed=int(getattr(args, "data_seed", 42)),
            max_train=getattr(args, "max_train", None),
            max_eval=getattr(args, "val_max_samples", None)
            if getattr(args, "val_max_samples", 0)
            else None,
            lfw_root=getattr(args, "lfw_root", None),
            celeba_root=getattr(args, "celeba_root", None),
        )
        train_paths = data["train"]
        eval_paths = data["eval"]
        split_stats = data["split_stats"]
    else:
        train_paths = list(train_paths)
        eval_paths = list(eval_paths or [])

    train_loader = make_face_dataloader(
        train_paths,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        num_workers=int(getattr(args, "num_workers", 0)),
    )
    val_loader = None
    if eval_paths and getattr(args, "early_stopping", True):
        val_loader = make_face_dataloader(
            eval_paths,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            num_workers=int(getattr(args, "num_workers", 0)),
        )

    # ---- FRs ----
    fr_preset = getattr(args, "fr_preset", "small")
    fr_models = getattr(args, "fr_models", None)
    if fr_models is None and getattr(args, "smoke_fr", False):
        fr_models = ",".join(SMOKE_FR_NAMES)
    train_fr_names = resolve_train_fr_names(preset=fr_preset, fr_models=fr_models)
    eval_fr_names = resolve_eval_fr_names(getattr(args, "eval_fr_models", None))
    # Smoke FaceNets may overlap default eval names — drop overlap from eval.
    overlap = set(train_fr_names) & set(eval_fr_names)
    if overlap and getattr(args, "smoke_fr", False):
        eval_fr_names = tuple(n for n in eval_fr_names if n not in overlap)
    else:
        assert_train_eval_disjoint(train_fr_names, eval_fr_names)

    frs = load_fr_ensemble(train_fr_names, device=device, keep_on_device=False)
    eval_frs = (
        load_fr_ensemble(eval_fr_names, device=device, keep_on_device=False)
        if eval_fr_names and val_loader is not None
        else []
    )

    # ---- module / opt ----
    module = DTCWTMetaMaskModule(
        channels=3,
        img_size=image_size,
        J=J,
        editable_levels=editable,
        init_std=init_std,
        epsilon=epsilon,
    ).to(device)

    opt: Optional[torch.optim.Optimizer] = None
    if optimizer_name == "adam":
        opt = torch.optim.Adam(module.parameters(), lr=lr)
    elif optimizer_name != "protego":
        raise ValueError(f"optimizer must be adam|protego, got {optimizer_name!r}")

    percep_w = 1.0
    start_epoch = 0
    global_step = 0
    elapsed_offset = 0.0
    best_score: Optional[float] = None
    best_epoch = -1
    patience = int(getattr(args, "patience", 2))
    patience_left = patience
    es_metric = str(getattr(args, "early_stopping_metric", "cos"))
    min_delta = float(getattr(args, "min_delta", 0.005))
    early_stopping = bool(getattr(args, "early_stopping", True))
    ckpt_every = int(getattr(args, "ckpt_every", 1000))
    preview_every = int(getattr(args, "preview_every", 100))
    max_steps = getattr(args, "max_steps", None)

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        module.load_meta_mask_state(ckpt["meta_mask"])
        if opt is not None and "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        percep_w = float(ckpt.get("percep_loss_weight", 1.0))
        elapsed_offset = float(ckpt.get("elapsed_seconds", 0.0))
        print(f"[resume] {resume_path} epoch={start_epoch} step={global_step}")

    # ---- config / logs ----
    first_batch = next(iter(train_loader))
    x0 = first_batch[0]
    batch_meta = observe_batch_hw(x0)
    native_hw = sample_native_hw(train_paths, max_samples=8)
    config = build_split_config_block(
        split_stats or {"n_train_total": len(train_paths), "n_eval_total": len(eval_paths)},
        image_size=image_size,
        data_seed=int(getattr(args, "data_seed", 42)),
        celeba_train_frac=float(getattr(args, "celeba_train_frac", 0.5)),
        dataset_native_hw_sample=native_hw,
    )
    config.update(
        {
            "roi_compose": roi_compose,
            "use_face_mask": use_face_mask,
            "mask_mode": mask_mode,
            "mask_shape": mask_shape,
            "mask_regions": list(mask_regions),
            "ssim_region": ssim_region,
            "optimizer": optimizer_name,
            "lr": lr,
            "epsilon": epsilon,
            "min_ssim": min_ssim,
            "editable_levels": list(editable),
            "J": J,
            "fr_preset": fr_preset,
            "train_fr_names": list(train_fr_names),
            "eval_fr_names": list(eval_fr_names),
            "batch_hw": batch_meta["batch_hw"],
            "epochs": epochs,
            "early_stopping": early_stopping,
            "early_stopping_metric": es_metric,
        }
    )
    # Rough ROI coverage on first batch (fixed / no-mask only without detector).
    with torch.no_grad():
        m0 = resolve_face_mask_by_size(
            x0.shape[0],
            list(first_batch[1]),
            torch.device("cpu"),
            image_size=image_size,
            use_face_mask=use_face_mask,
            mask_mode="fixed" if mask_mode == "landmarks" else mask_mode,
            mask_regions=mask_regions,
            mask_shape=mask_shape if mask_shape != "band" else "ellipse",
        )
        config["mask_coverage_mean"] = float(m0.mean().cpu())
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_log = out_dir / "train_log.csv"
    write_header = not (resume_path and train_log.exists())
    log_f = open(train_log, "a", newline="", encoding="utf-8")
    log_fields = [
        "epoch",
        "step",
        "loss",
        "cos_similarity",
        "ssim",
        "percep_term",
        "percep_weight",
        "linf_max",
        "n_frs",
        "optimizer",
        "lr",
        "image_size",
        "batch_h",
        "batch_w",
        "roi_compose",
        "mask_mode",
        "mask_shape",
        "mask_regions",
        "seconds",
    ]
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    if write_header:
        writer.writeheader()

    val_log_f = None
    val_writer = None
    if early_stopping and val_loader is not None:
        val_log = out_dir / "val_log.csv"
        vh = not (resume_path and val_log.exists())
        val_log_f = open(val_log, "a", newline="", encoding="utf-8")
        val_writer = csv.DictWriter(
            val_log_f,
            fieldnames=[
                "epoch",
                "step",
                "cos_mean",
                "euclid_mean",
                "ssim_mean",
                "score",
                "is_best",
                "image_size",
                "roi_compose",
            ],
        )
        if vh:
            val_writer.writeheader()

    print(
        f"[train-v2] out={out_dir} opt={optimizer_name} lr={lr} "
        f"levels={editable} roi={roi_compose} mask={mask_mode}/{mask_shape}/{mask_regions} "
        f"FRs={train_fr_names} size={image_size} batch_hw={batch_meta['batch_hw']}"
    )

    landmark_detector = None
    need_landmarks = use_face_mask and (
        mask_mode == "landmarks" or ssim_region == "full-landmarks"
    )
    if need_landmarks:
        try:
            from models.face_detector import get_default_detector

            landmark_detector = get_default_detector()
        except Exception as exc:  # pragma: no cover
            print(f"[warn] face detector unavailable ({exc}); using fixed ellipse fallback")
            if mask_mode == "landmarks":
                mask_mode = "fixed"

    t0 = time.time()
    last_row: Optional[dict[str, Any]] = None
    stopped_early = False

    try:
        for epoch in range(start_epoch, epochs):
            module.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
            for batch in pbar:
                if max_steps is not None and global_step >= int(max_steps):
                    stopped_early = True
                    break
                if len(batch) == 3:
                    x, paths, _src_hw = batch
                else:
                    x, paths = batch
                x = x.to(device)
                paths = list(paths)

                landmarks_batch = None
                if need_landmarks and landmark_detector is not None:
                    landmarks_batch = detect_landmarks_batch(paths, landmark_detector)

                face_mask = resolve_face_mask_by_size(
                    x.shape[0],
                    paths,
                    device,
                    image_size=image_size,
                    use_face_mask=use_face_mask,
                    mask_mode=mask_mode,
                    mask_regions=mask_regions,
                    mask_shape=mask_shape,
                    detector=landmark_detector,
                    landmarks_batch=landmarks_batch,
                )

                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                else:
                    module.zero_grad(set_to_none=True)

                n_e = float(len(frs))
                offload = len(frs) > 1
                sum_cos = 0.0
                total_f = 0.0
                ssim_f = 0.0
                percep_f = 0.0
                linf_max = 0.0
                x_prot_log = None

                # Recompute forward per FR (Phase 3.3): avoids retain_graph / VRAM blowup.
                for i, fr in enumerate(frs):
                    fr.to(device)
                    x_prot = module(x, face_mask=face_mask, roi_compose=roi_compose)
                    linf_max = max(
                        linf_max,
                        float((x_prot.detach() - x.detach()).abs().amax().cpu()),
                    )
                    with torch.no_grad():
                        f0 = fr(x)
                    f1 = fr(x_prot)
                    loss_id, cos_det = identity_cosine_loss(f0, f1)
                    fr_loss = loss_id / n_e
                    if i == 0:
                        ssim_mask = resolve_ssim_mask_by_size(
                            paths,
                            device,
                            image_size=image_size,
                            ssim_region=ssim_region,
                            face_mask=face_mask,
                            use_face_mask=use_face_mask,
                            landmarks_batch=landmarks_batch,
                            detector=landmark_detector,
                        )
                        percep_term, ssim_value = ssim_hinge_loss(
                            x_prot,
                            x,
                            min_ssim=min_ssim,
                            weight=percep_w,
                            ssim_mask=ssim_mask,
                        )
                        (fr_loss + percep_term).backward()
                        ssim_f = float(ssim_value.detach().cpu())
                        percep_f = float(percep_term.detach().cpu())
                        total_f += percep_f
                        x_prot_log = x_prot.detach()
                    else:
                        fr_loss.backward()
                    total_f += float(fr_loss.detach().cpu())
                    sum_cos += float(cos_det.cpu())
                    if offload and device.type == "cuda":
                        fr.cpu()
                        torch.cuda.empty_cache()

                if optimizer_name == "protego":
                    protego_sign_step(module, lr)
                else:
                    assert opt is not None
                    opt.step()

                percep_w = adapt_percep_weight(ssim_f, min_ssim, percep_w)
                cos_sim_f = sum_cos / n_e
                global_step += 1
                elapsed = elapsed_offset + (time.time() - t0)
                bh, bw = x.shape[-2], x.shape[-1]
                last_row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": total_f,
                    "cos_similarity": cos_sim_f,
                    "ssim": ssim_f,
                    "percep_term": percep_f,
                    "percep_weight": percep_w,
                    "linf_max": linf_max,
                    "n_frs": len(frs),
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "image_size": image_size,
                    "batch_h": bh,
                    "batch_w": bw,
                    "roi_compose": roi_compose,
                    "mask_mode": mask_mode,
                    "mask_shape": mask_shape,
                    "mask_regions": ",".join(mask_regions),
                    "seconds": elapsed,
                }
                writer.writerow(last_row)
                log_f.flush()
                pbar.set_postfix(
                    loss=f"{total_f:.3f}",
                    cos=f"{cos_sim_f:.3f}",
                    ssim=f"{ssim_f:.3f}",
                    w=f"{percep_w:.3g}",
                    opt=optimizer_name,
                )

                if preview_every > 0 and global_step % preview_every == 0 and x_prot_log is not None:
                    _save_preview(
                        x.detach().cpu(),
                        x_prot_log.cpu(),
                        face_mask.detach().cpu(),
                        preview_dir / f"epoch_{epoch+1:03d}_step_{global_step:06d}.jpg",
                    )

                if ckpt_every > 0 and global_step % ckpt_every == 0:
                    _atomic_torch_save(
                        _ckpt_payload(
                            module,
                            opt,
                            config,
                            epoch=epoch,  # 0-based; resume continues this epoch
                            global_step=global_step,
                            percep_w=percep_w,
                            last_row=last_row,
                            elapsed=elapsed,
                            optimizer_name=optimizer_name,
                            editable=editable,
                            image_size=image_size,
                            J=J,
                        ),
                        out_dir / CHECKPOINT_LATEST,
                    )

            if stopped_early:
                break

            # end-of-epoch validation / early stopping
            if early_stopping and val_loader is not None:
                metrics = validate_v2(
                    module,
                    frs,
                    val_loader,
                    device,
                    image_size=image_size,
                    roi_compose=roi_compose,
                    use_face_mask=use_face_mask,
                    mask_mode=mask_mode,
                    mask_regions=mask_regions,
                    mask_shape=mask_shape,
                    ssim_region=ssim_region,
                    landmark_detector=landmark_detector,
                    max_samples=int(getattr(args, "val_max_samples", 200) or 200),
                )
                score = early_stopping_score_v2(metrics, es_metric)
                improved = best_score is None or is_improvement_v2(
                    score, best_score, es_metric, min_delta
                )
                if improved:
                    best_score = score
                    best_epoch = epoch + 1
                    patience_left = patience
                    _atomic_torch_save(
                        {
                            "meta_mask": module.meta_mask_state(),
                            "config": config,
                            "editable_levels": list(editable),
                            "img_size": image_size,
                            "J": J,
                            "roi_compose": roi_compose,
                            "mask_mode": mask_mode,
                            "mask_shape": mask_shape,
                            "mask_regions": list(mask_regions),
                            "ssim_region": ssim_region,
                            "use_face_mask": use_face_mask,
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "val_metrics": metrics,
                        },
                        out_dir / META_MASK_BEST,
                    )
                else:
                    patience_left -= 1
                if val_writer is not None:
                    val_writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "cos_mean": metrics["cos_mean"],
                            "euclid_mean": metrics["euclid_mean"],
                            "ssim_mean": metrics["ssim_mean"],
                            "score": score,
                            "is_best": int(improved),
                            "image_size": image_size,
                            "roi_compose": roi_compose,
                        }
                    )
                    val_log_f.flush()
                print(
                    f"[val] epoch={epoch+1} cos={metrics['cos_mean']:.4f} "
                    f"ssim={metrics['ssim_mean']:.4f} score={score:.4f} "
                    f"best={best_score:.4f} patience={patience_left}"
                )
                if patience_left <= 0:
                    print("[early-stop] patience exhausted")
                    stopped_early = True
                    break

            # rolling ckpt at epoch end → next epoch index for resume
            _atomic_torch_save(
                _ckpt_payload(
                    module,
                    opt,
                    config,
                    epoch=min(epoch + 1, epochs),
                    global_step=global_step,
                    percep_w=percep_w,
                    last_row=last_row,
                    elapsed=elapsed_offset + (time.time() - t0),
                    optimizer_name=optimizer_name,
                    editable=editable,
                    image_size=image_size,
                    J=J,
                ),
                out_dir / CHECKPOINT_LATEST,
            )

    finally:
        log_f.close()
        if val_log_f is not None:
            val_log_f.close()

    _atomic_torch_save(
        {
            "meta_mask": module.meta_mask_state(),
            "config": config,
            "editable_levels": list(editable),
            "img_size": image_size,
            "J": J,
            "roi_compose": roi_compose,
            "mask_mode": mask_mode,
            "mask_shape": mask_shape,
            "mask_regions": list(mask_regions),
            "ssim_region": ssim_region,
            "use_face_mask": use_face_mask,
            "epoch": epochs,
            "global_step": global_step,
        },
        out_dir / META_MASK_FINAL,
    )

    summary = {
        "out_dir": str(out_dir),
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stopped_early": stopped_early,
        "roi_compose": roi_compose,
        "optimizer": optimizer_name,
        "image_size": image_size,
        "batch_hw": batch_meta["batch_hw"],
        "split_stats": split_stats,
        "last_row": last_row,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train-v2] done → {out_dir}")
    return out_dir


def _ckpt_payload(
    module,
    opt,
    config,
    *,
    epoch,
    global_step,
    percep_w,
    last_row,
    elapsed,
    optimizer_name,
    editable,
    image_size,
    J,
) -> dict:
    payload = {
        "meta_mask": module.meta_mask_state(),
        "module_state_dict": {
            k: v.detach().cpu()
            for k, v in module.state_dict().items()
            if k.startswith("delta_")
        },
        "config": config,
        "editable_levels": list(editable),
        "img_size": image_size,
        "J": J,
        "roi_compose": config.get("roi_compose", "pixel"),
        "mask_mode": config.get("mask_mode"),
        "mask_shape": config.get("mask_shape"),
        "mask_regions": config.get("mask_regions"),
        "ssim_region": config.get("ssim_region"),
        "use_face_mask": config.get("use_face_mask"),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "percep_loss_weight": float(percep_w),
        "optimizer": optimizer_name,
        "elapsed_seconds": float(elapsed),
        "last_row": last_row,
    }
    if opt is not None:
        payload["optimizer_state_dict"] = opt.state_dict()
    return payload
