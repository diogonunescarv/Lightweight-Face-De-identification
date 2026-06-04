import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def get_face_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Retorna a máscara facial elíptica (suave) para as imagens.
        """

        _, _, _, face_mask = self._get_buffers(device)
        return face_mask.expand(batch_size, -1, -1, -1)   # [B, 1, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        device = x.device

        # CÓDIGO ANTERIOR:
        # dct_delta = self.reconstruct_dct_delta(b, device)
        # photo_delta = self.reconstruct_photo_delta(b, device)
        # flow = self.reconstruct_flow(b, device)
        #
        # y = x + dct_delta
        # y = y.clamp(0.0, 1.0)
        #
        # y = self.warp(y, flow)
        #
        # y = y + photo_delta
        # y = y.clamp(0.0, 1.0)

        y = x

        # DCT
        if not self.disable_dct:
            dct_delta = self.reconstruct_dct_delta(b, device)
            y = y + dct_delta
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