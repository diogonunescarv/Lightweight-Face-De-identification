import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from models.embedders import normalized_embedding

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

def ssim_index_masked(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                      kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    SSIM médio apenas na região indicada pela máscara.
    x, y: [B, C, H, W] em [0,1]
    mask: [B, 1, H, W] ou [1, 1, H, W] com valores em [0,1].
    Retorna [B].
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
    )   # [B, C, H, W]

    # Expande a máscara para o mesmo número de canais
    mask_expanded = mask.expand_as(ssim_map)   # [B, C, H, W]

    # Média ponderada sobre todos os pixels e canais
    weighted_ssim = (ssim_map * mask_expanded).sum(dim=(1,2,3)) / (mask_expanded.sum(dim=(1,2,3)) + 1e-8)
    return weighted_ssim   # [B]

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