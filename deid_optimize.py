"""
deid_optimize.py

Otimização offline de uma transformação facial leve para desidentificação visualmente preservada.

A transformação T_theta(x) combina:
  1. perturbação DCT de baixa/média frequência;
  2. microdeformação geométrica suave;
  3. ajuste fotométrico local suave.

Durante o treinamento, embeddings faciais são usados apenas como avaliadores.
Na aplicação final, o transformador é puramente determinístico e leve.

Uso esperado:
  Treinamento:
    python deid_optimize.py train \
      --data ./faces_train \
      --out ./runs/deid_v1 \
      --image-size 224 \
      --steps 2000 \
      --batch-size 8 \
      --device cpu

  Aplicação:
    python deid_optimize.py apply \
      --checkpoint ./runs/deid_v1/transform.pt \
      --input ./faces_test \
      --output ./faces_deid \
      --image-size 224 \
      --device cpu
"""

from __future__ import annotations

import argparse
import math
from sched import scheduler
from sched import scheduler
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceImageFolder(Dataset):
    def __init__(self, root: str | Path, image_size: int):
        self.root = Path(root)
        self.image_size = int(image_size)

        if not self.root.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {self.root}")

        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        )

        if not self.files:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return tensor, str(path)


class WarmupScheduler:
    """Scheduler que combina warmup linear com outro scheduler."""
    def __init__(self, optimizer, base_scheduler, warmup_steps, init_lr):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            # Warmup linear: de 0 até init_lr
            factor = self.step_num / self.warmup_steps
            lr = self.init_lr * factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Após warmup, usa o scheduler base
            self.base_scheduler.step()

# ============================================================
# Utilidades de imagem
# ============================================================

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    x: tensor [3,H,W] em [0,1].
    """
    x = x.detach().cpu().clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_preview(original: torch.Tensor, transformed: torch.Tensor, path: Path, max_items: int = 4):
    """
    Salva uma imagem lado-a-lado: original | transformada.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    b = min(original.shape[0], max_items)
    rows = []

    for i in range(b):
        o = tensor_to_pil(original[i])
        t = tensor_to_pil(transformed[i])

        canvas = Image.new("RGB", (o.width + t.width, o.height), (255, 255, 255))
        canvas.paste(o, (0, 0))
        canvas.paste(t, (o.width, 0))
        rows.append(canvas)

    out = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), (255, 255, 255))
    for i, r in enumerate(rows):
        out.paste(r, (0, i * r.height))

    out.save(path)


def build_elliptic_face_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Máscara elíptica simples para evitar modificar excessivamente o fundo.
    Assume faces aproximadamente alinhadas/cortadas.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )

    # Elipse levemente verticalizada.
    val = (xx / 0.82) ** 2 + ((yy + 0.03) / 0.95) ** 2
    mask = torch.exp(-4.0 * torch.clamp(val - 0.65, min=0.0))
    mask = mask.clamp(0.0, 1.0)

    return mask[None, None, :, :]


# ============================================================
# DCT truncada
# ============================================================

def dct_basis(num_freq: int, signal_size: int, device: torch.device) -> torch.Tensor:
    """
    Base ortonormal DCT-II.

    Retorna matriz B [num_freq, signal_size], onde:
      basis[u, x]
    """
    n = torch.arange(signal_size, device=device).float()
    basis = []

    for k in range(num_freq):
        if k == 0:
            scale = math.sqrt(1.0 / signal_size)
        else:
            scale = math.sqrt(2.0 / signal_size)

        row = scale * torch.cos(math.pi * (n + 0.5) * k / signal_size)
        basis.append(row)

    return torch.stack(basis, dim=0)


def build_dct_frequency_mask(k: int, fmin: int, fmax: int, device: torch.device) -> torch.Tensor:
    """
    Máscara para manter frequências baixas/médias.

    Exclui DC e permite coeficientes com:
      fmin <= u + v <= fmax
    """
    uu, vv = torch.meshgrid(
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        indexing="ij",
    )

    radius = uu + vv
    mask = (radius >= fmin) & (radius <= fmax)
    mask[0, 0] = False

    return mask.float()[None, None, :, :]


# ============================================================
# SSIM diferenciável
# ============================================================

def gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device):
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()

    kernel = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
    return kernel


def ssim_index(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Retorna SSIM médio por imagem: [B].
    Espera x,y em [0,1], shape [B,C,H,W].
    """
    c = x.shape[1]
    device = x.device

    kernel = gaussian_kernel(kernel_size, sigma, c, device)
    padding = kernel_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=c) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )

    return ssim_map.mean(dim=(1, 2, 3))


# ============================================================
# Transformador parametrizado
# ============================================================

class LightweightDeIdentifier(nn.Module):
    """
    T_theta(x) = P_theta(G_theta(W_theta(x)))

    Componentes:
      - DCT baixa/média frequência;
      - campo de deformação suave de baixa resolução;
      - ajuste fotométrico local suave.
    """

    def __init__(
        self,
        image_size: int = 224,
        dct_k: int = 32,
        dct_fmin: int = 2,
        dct_fmax: int = 18,
        flow_grid: int = 12,
        photo_grid: int = 12,
        max_dct_amp: float = 0.035,
        max_flow_px: float = 2.0,
        max_photo_amp: float = 0.035,
        use_face_mask: bool = True,
        disable_dct: bool = False,
        disable_flow: bool = False,
        disable_photo: bool = False,
    ):
        super().__init__()

        self.image_size = int(image_size)
        self.dct_k = int(dct_k)
        self.dct_fmin = int(dct_fmin)
        self.dct_fmax = int(dct_fmax)
        self.flow_grid = int(flow_grid)
        self.photo_grid = int(photo_grid)

        self.max_dct_amp = float(max_dct_amp)
        self.max_flow_px = float(max_flow_px)
        self.max_photo_amp = float(max_photo_amp)
        self.use_face_mask = bool(use_face_mask)

        self.disable_dct = disable_dct
        self.disable_flow = disable_flow
        self.disable_photo = disable_photo

        # Coeficientes DCT universais por canal.
        self.raw_dct = nn.Parameter(torch.zeros(1, 3, self.dct_k, self.dct_k))

        # Campo de fluxo universal em baixa resolução.
        # Canal 0 = deslocamento horizontal em pixels.
        # Canal 1 = deslocamento vertical em pixels.
        self.raw_flow = nn.Parameter(torch.zeros(1, 2, self.flow_grid, self.flow_grid))

        # Ajuste fotométrico local universal em baixa resolução.
        self.raw_photo = nn.Parameter(torch.zeros(1, 1, self.photo_grid, self.photo_grid))

        # Pequeno viés por canal.
        self.raw_channel_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def _get_buffers(self, device: torch.device):
        h = w = self.image_size

        bh = dct_basis(self.dct_k, h, device)
        bw = dct_basis(self.dct_k, w, device)
        dct_mask = build_dct_frequency_mask(self.dct_k, self.dct_fmin, self.dct_fmax, device)

        if self.use_face_mask:
            face_mask = build_elliptic_face_mask(h, w, device)
        else:
            face_mask = torch.ones(1, 1, h, w, device=device)

        return bh, bw, dct_mask, face_mask

    def reconstruct_dct_delta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        bh, bw, dct_mask, face_mask = self._get_buffers(device)

        coeff = self.max_dct_amp * torch.tanh(self.raw_dct) * dct_mask
        coeff = coeff.expand(batch_size, -1, -1, -1)

        # delta[b,c,h,w] = sum_{u,v} coeff[b,c,u,v] * B[u,h] * B[v,w]
        delta = torch.einsum("uh,bcuv,vw->bchw", bh, coeff, bw)
        delta = delta * face_mask

        return delta

    def reconstruct_photo_delta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        h = w = self.image_size
        _, _, _, face_mask = self._get_buffers(device)

        photo = self.max_photo_amp * torch.tanh(self.raw_photo)
        photo = F.interpolate(photo, size=(h, w), mode="bicubic", align_corners=False)
        photo = photo.expand(batch_size, 3, h, w)

        bias = 0.015 * torch.tanh(self.raw_channel_bias)
        photo = photo + bias

        return photo * face_mask

    def reconstruct_flow(self, batch_size: int, device: torch.device) -> torch.Tensor:
        h = w = self.image_size
        _, _, _, face_mask = self._get_buffers(device)

        flow = self.max_flow_px * torch.tanh(self.raw_flow)
        flow = F.interpolate(flow, size=(h, w), mode="bicubic", align_corners=False)
        flow = flow.expand(batch_size, 2, h, w)

        return flow * face_mask

    def warp(self, x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )

        base_grid = torch.stack([xx, yy], dim=-1)
        base_grid = base_grid[None, :, :, :].expand(b, h, w, 2)

        dx = 2.0 * flow_px[:, 0, :, :] / max(w - 1, 1)
        dy = 2.0 * flow_px[:, 1, :, :] / max(h - 1, 1)

        grid = base_grid.clone()
        grid[:, :, :, 0] = grid[:, :, :, 0] + dx
        grid[:, :, :, 1] = grid[:, :, :, 1] + dy

        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        device = x.device

        # CÓDIGO ANTERIOR:
        # dct_delta = self.reconstruct_dct_delta(b, device)
        # photo_delta = self.reconstruct_photo_delta(b, device)
        # flow = self.reconstruct_flow(b, device)

        # DCT
        if not self.disable_dct:
            dct_delta = self.reconstruct_dct_delta(b, device)
            y = x + dct_delta
            y = y.clamp(0.0, 1.0)
    
        # Warp (flow)
        if not self.disable_flow:    
            flow = self.reconstruct_flow(b, device)
            y = self.warp(y, flow)

        # Photo adjustment
        if not self.disable_photo:
            photo_delta = self.reconstruct_photo_delta(b, device)
            y = y + photo_delta
            y = y.clamp(0.0, 1.0)

        return y

    def regularization(self) -> dict:
        flow = torch.tanh(self.raw_flow)

        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

        flow_smooth = (dx ** 2).mean() + (dy ** 2).mean()

        dct_l2 = (torch.tanh(self.raw_dct) ** 2).mean()
        photo_l2 = (torch.tanh(self.raw_photo) ** 2).mean()

        return {
            "flow_smooth": flow_smooth,
            "dct_l2": dct_l2,
            "photo_l2": photo_l2,
        }


# ============================================================
# Interface para embeddings autorizados
# ============================================================

def load_authorized_embedders(device: torch.device) -> List[nn.Module]:
    """
    Espera um arquivo local face_embedder.py com:

        def build_face_embedders(device):
            ...
            return [model1, model2, ...]

    Cada modelo deve:
      - receber tensor [B,3,H,W] em [0,1];
      - retornar embedding [B,D];
      - ser diferenciável em relação à imagem de entrada;
      - estar autorizado para uso no projeto.
    """
    try:
        from face_embedder import build_face_embedders
    except ImportError as exc:
        raise RuntimeError(
            "Arquivo face_embedder.py não encontrado. "
            "Crie esse arquivo com a função build_face_embedders(device)."
        ) from exc

    embedders = build_face_embedders(device)

    if not isinstance(embedders, list) or len(embedders) == 0:
        raise RuntimeError("build_face_embedders(device) deve retornar uma lista não vazia de modelos.")

    for model in embedders:
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    return embedders


def normalized_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    emb = model(x)
    emb = F.normalize(emb, dim=1)
    return emb


# ============================================================
# Loss de treinamento
# ============================================================

def identity_loss_ensemble(
    embedders: List[nn.Module],
    x_original: torch.Tensor,
    x_transformed: torch.Tensor,
    target_cos: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimiza a similaridade cosseno entre x e T(x).
    Usa hinge:
        max(0, cos(f(T(x)), f(x)) - target_cos)

    target_cos mais baixo => desidentificação mais agressiva.
    """
    losses = []
    sims = []

    for model in embedders:
        with torch.no_grad():
            e0 = normalized_embedding(model, x_original)

        e1 = normalized_embedding(model, x_transformed)

        cos = (e0 * e1).sum(dim=1)
        loss = F.relu(cos - target_cos).mean()

        losses.append(loss)
        sims.append(cos.detach())

    loss_all = torch.stack(losses).mean()
    sim_all = torch.cat(sims).mean()

    return loss_all, sim_all


def pixel_l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).mean()


def total_variation_loss(y: torch.Tensor) -> torch.Tensor:
    dx = y[:, :, :, 1:] - y[:, :, :, :-1]
    dy = y[:, :, 1:, :] - y[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


# ============================================================
# Treinamento
# ============================================================

def train(args):
    device = torch.device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "steps.csv"
    with open(log_path, "w") as f:
        f.write("step,loss,id,cos,ssim,pix,tv,elapsed,lr,euclid\n")

    dataset = FaceImageFolder(args.data, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    embedders = load_authorized_embedders(device)

    transformer = LightweightDeIdentifier(
        image_size=args.image_size,
        dct_k=args.dct_k,
        dct_fmin=args.dct_fmin,
        dct_fmax=args.dct_fmax,
        flow_grid=args.flow_grid,
        photo_grid=args.photo_grid,
        max_dct_amp=args.max_dct_amp,
        max_flow_px=args.max_flow_px,
        max_photo_amp=args.max_photo_amp,
        use_face_mask=not args.no_face_mask,
        disable_dct=args.disable_dct,
        disable_flow=args.disable_flow,
        disable_photo=args.disable_photo,
    ).to(device)

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
            pct_start=warmup_steps / args.steps if warmup_steps > 0 else 0.1,
            anneal_strategy='cos'
        )
        print(f"Usando OneCycleLR: max_lr={args.lr}, total_steps={args.steps}, "
              f"pct_start={warmup_steps/args.steps if warmup_steps>0 else 0.1}")
    else:
        print("Usando LR fixo (sem scheduler)")

    step = 0
    start = time.time()

    while step < args.steps:
        for x, _paths in loader:
            if step >= args.steps:
                break

            x = x.to(device)

            y = transformer(x)

            loss_id, mean_cos = identity_loss_ensemble(
                embedders=embedders,
                x_original=x,
                x_transformed=y,
                target_cos=args.target_cos,
            )

            ssim = ssim_index(x, y)
            loss_ssim = F.relu(args.tau_ssim - ssim).mean()

            loss_pix = pixel_l2_loss(x, y)
            loss_tv = total_variation_loss(y)

            regs = transformer.regularization()

            loss = (
                args.lambda_id * loss_id
                + args.lambda_ssim * loss_ssim
                + args.lambda_pixel * loss_pix
                + args.lambda_tv * loss_tv
                + args.lambda_flow_smooth * regs["flow_smooth"]
                + args.lambda_dct_l2 * regs["dct_l2"]
                + args.lambda_photo_l2 * regs["photo_l2"]
            )

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
                        f"{ssim.mean().item():.4f}," 
                        f"{loss_pix.item():.6f}," 
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
                    f"ssim={ssim.mean().item():.4f} "
                    f"pix={loss_pix.item():.6f} "
                    f"tv={loss_tv.item():.6f} "
                    f"elapsed={elapsed:.1f}s "
                    f"lr={current_lr:.6f}\n"
                )

            if step % args.preview_every == 0:
                save_preview(x, y, preview_dir / f"step_{step:06d}.jpg")

            if step % args.save_every == 0 and step > 0:
                save_checkpoint(transformer, args, out_dir / "transform.pt", step)

            step += 1

    save_checkpoint(transformer, args, out_dir / "transform.pt", step)
    print(f"Modelo salvo em: {out_dir / 'transform.pt'}")
    print(f"Log salvo em: {log_path}")


def save_checkpoint(transformer: LightweightDeIdentifier, args, path: Path, step: int):
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
        "state_dict": transformer.state_dict(),
        "disable_dct": args.disable_dct,
        "disable_flow": args.disable_flow,
        "disable_photo": args.disable_photo,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


# ============================================================
# Aplicação da transformação final
# ============================================================

def load_transformer_from_checkpoint(path: str | Path, device: torch.device) -> LightweightDeIdentifier:
    ckpt = torch.load(path, map_location=device)

    transformer = LightweightDeIdentifier(
        image_size=ckpt["image_size"],
        dct_k=ckpt["dct_k"],
        dct_fmin=ckpt["dct_fmin"],
        dct_fmax=ckpt["dct_fmax"],
        flow_grid=ckpt["flow_grid"],
        photo_grid=ckpt["photo_grid"],
        max_dct_amp=ckpt["max_dct_amp"],
        max_flow_px=ckpt["max_flow_px"],
        max_photo_amp=ckpt["max_photo_amp"],
        use_face_mask=ckpt["use_face_mask"],
        disable_dct=ckpt["disable_dct"],
        disable_flow=ckpt["disable_flow"],
        disable_photo=ckpt["disable_photo"],
    ).to(device)

    transformer.load_state_dict(ckpt["state_dict"])
    transformer.eval()

    return transformer


@torch.no_grad()
def apply_transform(args):
    device = torch.device(args.device)

    transformer = load_transformer_from_checkpoint(args.checkpoint, device)
    dataset = FaceImageFolder(args.input, args.image_size)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    total = 0
    start = time.time()

    for x, paths in loader:
        x = x.to(device)

        t0 = time.time()
        y = transformer(x)
        batch_time = time.time() - t0

        for img_tensor, src_path in zip(y, paths):
            src = Path(src_path)
            dst = output_dir / src.name
            tensor_to_pil(img_tensor).save(dst, quality=95)
            total += 1

        ms_per_image = 1000.0 * batch_time / max(x.shape[0], 1)
        print(f"Processadas {total} imagens | {ms_per_image:.2f} ms/imagem")

    elapsed = time.time() - start
    print(f"Concluído. Total: {total} imagens em {elapsed:.2f}s")

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)

    # Carrega o transformador
    transformer = load_transformer_from_checkpoint(args.checkpoint, device)
    transformer.eval()

    # Carrega os embedders (os mesmos usados no treino)
    embedders = load_authorized_embedders(device)

    # Dataset de validação (não o mesmo do treino)
    dataset = FaceImageFolder(args.data, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_cos = []
    all_ssim = []
    total_images = 0

    for x, paths in loader:
        x = x.to(device)
        y = transformer(x)

        # Similaridade cosseno (média sobre todos os embedders)
        batch_cos = []
        for model in embedders:
            e0 = F.normalize(model(x), dim=1)
            e1 = F.normalize(model(y), dim=1)
            cos = (e0 * e1).sum(dim=1)      # shape [B]
            batch_cos.append(cos)
        # Média sobre os embedders (se mais de um)
        mean_cos = torch.stack(batch_cos).mean(dim=0)   # [B]
        all_cos.append(mean_cos)

        # SSIM
        ssim_vals = ssim_index(x, y)   # [B]
        all_ssim.append(ssim_vals)

        total_images += x.shape[0]

    # Concatena os tensores de todos os batches
    cos_all = torch.cat(all_cos)        # [N]
    ssim_all = torch.cat(all_ssim)      # [N]

    mean_cos = cos_all.mean().item()
    mean_ssim = ssim_all.mean().item()
    # Distância euclidiana média no espaço de embeddings
    mean_euclid = math.sqrt(2.0 * (1.0 - mean_cos))

    print("\n========== Evaluation Results ==========")
    print(f"Total images evaluated: {total_images}")
    print(f"Mean cosine similarity: {mean_cos:.4f}")
    print(f"Mean Euclidean distance: {mean_euclid:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print("========================================\n")

    # Opcional: salvar os resultados em um arquivo
    if args.output_summary:
        out_path = Path(args.output_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(f"images,{total_images}\n")
            f.write(f"cos_mean,{mean_cos:.6f}\n")
            f.write(f"euclid_mean,{mean_euclid:.6f}\n")
            f.write(f"ssim_mean,{mean_ssim:.6f}\n")
        print(f"Summary saved to {out_path}")


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Otimização de transformação leve para desidentificação facial autorizada."
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # -------------------------
    # train
    # -------------------------
    p_train = sub.add_parser("train")

    p_train.add_argument("--data", required=True, help="Pasta com faces normalizadas/cortadas para treinamento.")
    p_train.add_argument("--out", required=True, help="Pasta de saída.")
    p_train.add_argument("--device", default="cpu", help="cpu ou cuda.")
    p_train.add_argument("--image-size", type=int, default=224)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--lr", type=float, default=2e-2)

    p_train.add_argument("--dct-k", type=int, default=32)
    p_train.add_argument("--dct-fmin", type=int, default=2)
    p_train.add_argument("--dct-fmax", type=int, default=18)
    p_train.add_argument("--flow-grid", type=int, default=12)
    p_train.add_argument("--photo-grid", type=int, default=12)

    p_train.add_argument("--max-dct-amp", type=float, default=0.035)
    p_train.add_argument("--max-flow-px", type=float, default=2.0)
    p_train.add_argument("--max-photo-amp", type=float, default=0.035)
    p_train.add_argument("--no-face-mask", action="store_true")

    p_train.add_argument("--target-cos", type=float, default=0.25)
    p_train.add_argument("--tau-ssim", type=float, default=0.95)

    p_train.add_argument("--lambda-id", type=float, default=1.0)
    p_train.add_argument("--lambda-ssim", type=float, default=20.0)
    p_train.add_argument("--lambda-pixel", type=float, default=2.0)
    p_train.add_argument("--lambda-tv", type=float, default=0.2)
    p_train.add_argument("--lambda-flow-smooth", type=float, default=5.0)
    p_train.add_argument("--lambda-dct-l2", type=float, default=0.01)
    p_train.add_argument("--lambda-photo-l2", type=float, default=0.01)

    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--preview-every", type=int, default=100)
    p_train.add_argument("--save-every", type=int, default=500)

    p_train.add_argument("--lr-scheduler", type=str, default="none",
                     choices=["none", "cosine", "step", "onecycle"],
                     help="Tipo de agendador: none (fixo), cosine, step, onecycle")

    p_train.add_argument("--lr-warmup-steps", type=int, default=0,
                     help="Número de passos para warmup linear (só cosine/onecycle)")
    p_train.add_argument("--lr-eta-min", type=float, default=0.0,
                        help="LR mínimo para cosine annealing")
    p_train.add_argument("--lr-step-size", type=int, default=500,
                        help="Intervalo de steps para StepLR")
    p_train.add_argument("--lr-gamma", type=float, default=0.5,
                        help="Fator de decaimento para StepLR")

    p_train.add_argument("--disable-dct", action="store_true", help="Desabilita perturbação DCT")
    p_train.add_argument("--disable-flow", action="store_true", help="Desabilita deformação geométrica")
    p_train.add_argument("--disable-photo", action="store_true", help="Desabilita ajuste fotométrico")

    # -------------------------
    # apply
    # -------------------------
    p_apply = sub.add_parser("apply")

    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--input", required=True)
    p_apply.add_argument("--output", required=True)
    p_apply.add_argument("--device", default="cpu")
    p_apply.add_argument("--image-size", type=int, default=224)
    p_apply.add_argument("--batch-size", type=int, default=8)
    p_apply.add_argument("--num-workers", type=int, default=0)

    # -------------------------
    # evaluate
    # -------------------------
    p_evaluate = sub.add_parser("evaluate", help="Avalia um checkpoint em um conjunto de validação")

    p_evaluate.add_argument("--checkpoint", required=True, help="Arquivo .pt do transformador treinado")
    p_evaluate.add_argument("--data", required=True, help="Pasta com imagens de validação (faces)")
    p_evaluate.add_argument("--device", default="cpu", help="cpu ou cuda")
    p_evaluate.add_argument("--image-size", type=int, default=224)
    p_evaluate.add_argument("--batch-size", type=int, default=8)
    p_evaluate.add_argument("--num-workers", type=int, default=0)
    p_evaluate.add_argument("--output-summary", type=str, default=None,
                            help="Opcional: caminho para salvar um resumo em CSV/txt")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "apply":
        apply_transform(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        raise RuntimeError(f"Modo desconhecido: {args.mode}")


if __name__ == "__main__":
    main()
