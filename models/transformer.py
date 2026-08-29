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
      - DCT baixa/média frequência ou DT-CWT;
      - campo de deformação suave de baixa resolução;
      - ajuste fotométrico local suave.
    """

    def __init__(
        self,
        image_size: int = 224,
        transform_type: str = "dct", # "dct" ou "dt-cwt"
        dct_k: int = 32,
        dct_fmin: int = 2,
        dct_fmax: int = 18,
        wavelet_J: int = 3,                    # níveis da wavelet
        max_wavelet_amp: float = 0.2,          # amplitude máxima para escala/fase
        flow_grid: int = 12,
        photo_grid: int = 12,
        max_dct_amp: float = 0.035,
        max_flow_px: float = 2.0,
        max_photo_amp: float = 0.035,
        use_face_mask: bool = True,
        mask_mode: str = "fixed",
        mask_regions: tuple[str, ...] = ("full",),
        disable_dct: bool = False,            # desabilita transformada espectral (DCT ou wavelet)
        disable_flow: bool = False,
        disable_photo: bool = False,
    ):
        super().__init__()

        self.transform_type = transform_type
        self.wavelet_J = wavelet_J
        self.max_wavelet_amp = max_wavelet_amp

        self.image_size = image_size
        self.dct_k = dct_k
        self.dct_fmin = dct_fmin
        self.dct_fmax = dct_fmax
        self.flow_grid = flow_grid
        self.photo_grid = photo_grid

        self.max_dct_amp = max_dct_amp
        self.max_flow_px = max_flow_px
        self.max_photo_amp = max_photo_amp
        self.use_face_mask = use_face_mask
        self.mask_mode = mask_mode
        self.mask_regions = mask_regions

        self.disable_dct = disable_dct
        self.disable_flow = disable_flow
        self.disable_photo = disable_photo

        # ----- Parâmetros DCT (sempre existem, mas só usados se transform_type == 'dct')
        self.raw_dct = nn.Parameter(torch.zeros(1, 3, self.dct_k, self.dct_k))

        # ----- Parâmetros DT-CWT (somente se transform_type == 'dtcwt') -----
        if transform_type == 'dtcwt':
            from .wavelet_transform import DTCWTUndecimated  # ou define no mesmo arquivo
            self.wavelet = DTCWTUndecimated(J=wavelet_J)
            # Parâmetros por nível (J) e orientação (6)
            self.raw_wavelet_mag_scale = nn.Parameter(torch.zeros(1, 6, wavelet_J))
            self.raw_wavelet_phase_shift = nn.Parameter(torch.zeros(1, 6, wavelet_J))
        else:
            self.wavelet = None
            self.raw_wavelet_mag_scale = None
            self.raw_wavelet_phase_shift = None

        # ----- Parâmetros comuns (flow, photo) ----
        
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

        return bh, bw, dct_mask

    def get_fixed_face_mask(self, device: torch.device) -> torch.Tensor:
        if self.use_face_mask:
            return build_elliptic_face_mask(self.image_size, self.image_size, device)
        return torch.ones(1, 1, self.image_size, self.image_size, device=device)

    def reconstruct_dct_delta(
        self, batch_size: int, device: torch.device, face_mask: torch.Tensor
    ) -> torch.Tensor:
        bh, bw, dct_mask = self._get_buffers(device)

        coeff = self.max_dct_amp * torch.tanh(self.raw_dct) * dct_mask
        coeff = coeff.expand(batch_size, -1, -1, -1)

        # delta[b,c,h,w] = sum_{u,v} coeff[b,c,u,v] * B[u,h] * B[v,w]
        delta = torch.einsum("uh,bcuv,vw->bchw", bh, coeff, bw)
        delta = delta * face_mask

        return delta

    def reconstruct_photo_delta(
        self, batch_size: int, device: torch.device, face_mask: torch.Tensor
    ) -> torch.Tensor:
        h = w = self.image_size

        photo = self.max_photo_amp * torch.tanh(self.raw_photo)
        photo = F.interpolate(photo, size=(h, w), mode="bicubic", align_corners=False)
        photo = photo.expand(batch_size, 3, h, w)

        # 0.015 de viés por canal (para evitar saturação) -> posso fazer isso ser um hiperparametro, mas por enquanto vou deixar fixo
        bias = 0.015 * torch.tanh(self.raw_channel_bias)
        photo = photo + bias

        return photo * face_mask

    def reconstruct_flow(
        self, batch_size: int, device: torch.device, face_mask: torch.Tensor
    ) -> torch.Tensor:
        h = w = self.image_size

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
        Retorna a máscara facial elíptica fixa (suave) para as imagens.
        """
        return self.get_fixed_face_mask(device).expand(batch_size, -1, -1, -1)

    def forward(self, x: torch.Tensor, face_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        device = x.device

        if face_mask is None:
            face_mask = self.get_face_mask(b, device)

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

        # Transformada espectral (DCT ou Wavelet)
        if not self.disable_dct:
            if self.transform_type == 'dct':
                dct_delta = self.reconstruct_dct_delta(b, device, face_mask)
                y = y + dct_delta
                y = y.clamp(0.0, 1.0)
            elif self.transform_type == 'dtcwt':
                Yl, Yh = self.wavelet(x)   # Yl: aproximação, Yh: lista de detalhes por nível
                Yh_perturbed = []
                for level, coeffs in enumerate(Yh):
                    # coeffs: [B, 6, C, H_l, W_l] (complexo)
                    mag = torch.abs(coeffs)
                    phase = torch.angle(coeffs)

                    # Parâmetros para este nível e orientação (broadcast)
                    scale = 1.0 + self.max_wavelet_amp * torch.tanh(
                        self.raw_wavelet_mag_scale[:, :, level].view(1, -1, 1, 1, 1)
                    )
                    phase_shift = self.max_wavelet_amp * torch.tanh(
                        self.raw_wavelet_phase_shift[:, :, level].view(1, -1, 1, 1, 1)
                    )

                    new_mag = mag * scale
                    new_phase = phase + phase_shift
                    new_coeffs = new_mag * torch.exp(1j * new_phase)
                    Yh_perturbed.append(new_coeffs)

                y_wavelet = self.wavelet.inverse(Yl, Yh_perturbed)
                y_wavelet = y_wavelet.clamp(0.0, 1.0)

                y = y_wavelet * face_mask + x * (1 - face_mask)
                y = y.clamp(0.0, 1.0)

        # Warp (flow)
        if not self.disable_flow:
            flow = self.reconstruct_flow(b, device, face_mask)
            y = self.warp(y, flow)

        # Photo adjustment
        if not self.disable_photo:
            photo_delta = self.reconstruct_photo_delta(b, device, face_mask)
            y = y + photo_delta
            y = y.clamp(0.0, 1.0)

        return y

    def regularization(self) -> dict:
        flow = torch.tanh(self.raw_flow)

        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

        flow_smooth = (dx ** 2).mean() + (dy ** 2).mean()

        photo_l2 = (torch.tanh(self.raw_photo) ** 2).mean()

        reg = {
            "flow_smooth": flow_smooth,
            "photo_l2": photo_l2,
        }

        if self.transform_type == 'dct':
            reg["dct_l2"] = (torch.tanh(self.raw_dct) ** 2).mean()
        elif self.transform_type == 'dtcwt' and self.raw_wavelet_mag_scale is not None:
            mag_l2 = (torch.tanh(self.raw_wavelet_mag_scale) ** 2).mean()
            phase_l2 = (torch.tanh(self.raw_wavelet_phase_shift) ** 2).mean()

            reg["wavelet_mag_l2"] = mag_l2
            reg["wavelet_phase_l2"] = phase_l2
            
            # Suavidade entre níveis
            smooth_mag = (self.raw_wavelet_mag_scale[:, :, 1:] - self.raw_wavelet_mag_scale[:, :, :-1]).abs().mean()
            
            reg["wavelet_mag_smooth"] = smooth_mag

        return reg