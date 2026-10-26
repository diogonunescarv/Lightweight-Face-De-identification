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

def parse_wavelet_level_amp_scales(
    scales: str | tuple[float, ...] | list[float] | None,
    wavelet_J: int,
) -> tuple[float, ...]:
    """Parse per-level amplitude multipliers; default all ones."""
    if scales is None or scales == "":
        return tuple(1.0 for _ in range(wavelet_J))
    if isinstance(scales, str):
        parts = [p.strip() for p in scales.split(",") if p.strip()]
        values = tuple(float(p) for p in parts)
    else:
        values = tuple(float(s) for s in scales)
    if len(values) != wavelet_J:
        raise ValueError(
            f"wavelet_level_amp_scales must have length wavelet_J={wavelet_J}, got {len(values)}"
        )
    return values


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
        transform_type: str = "dct",  # "dct" ou "dtcwt"
        dct_k: int = 32,
        dct_fmin: int = 2,
        dct_fmax: int = 18,
        wavelet_J: int = 3,
        max_wavelet_amp: float = 0.2,
        max_wavelet_mag_amp: float | None = None,
        max_wavelet_phase_amp: float | None = None,
        wavelet_level_amp_scales: str | tuple[float, ...] | list[float] | None = None,
        wavelet_polar_mode: str = "complex",
        wavelet_per_channel: bool = False,
        wavelet_spatial_grid: int = 0,
        wavelet_yl_gain: bool = False,
        max_wavelet_yl_amp: float = 0.05,
        flow_grid: int = 12,
        photo_grid: int = 12,
        max_dct_amp: float = 0.035,
        max_flow_px: float = 2.0,
        max_photo_amp: float = 0.035,
        use_face_mask: bool = True,
        mask_mode: str = "fixed",
        mask_regions: tuple[str, ...] = ("full",),
        mask_shape: str = "ellipse",
        disable_dct: bool = False,
        disable_flow: bool = False,
        disable_photo: bool = False,
    ):
        super().__init__()

        if wavelet_polar_mode not in ("complex", "legacy"):
            raise ValueError(
                f"wavelet_polar_mode must be 'complex' or 'legacy', got {wavelet_polar_mode!r}"
            )
        if wavelet_per_channel and wavelet_spatial_grid > 0:
            raise ValueError(
                "wavelet_per_channel and wavelet_spatial_grid>0 are mutually exclusive "
                "(enable one expressivity mode at a time)."
            )
        if wavelet_spatial_grid < 0:
            raise ValueError("wavelet_spatial_grid must be >= 0")
        if wavelet_polar_mode == "legacy" and (
            wavelet_per_channel or wavelet_spatial_grid > 0 or wavelet_yl_gain
        ):
            raise ValueError(
                "legacy wavelet_polar_mode only supports global (1,6,J) mag/phase params"
            )

        self.transform_type = transform_type
        self.wavelet_J = wavelet_J
        self.max_wavelet_amp = max_wavelet_amp
        self.max_wavelet_mag_amp = (
            max_wavelet_amp if max_wavelet_mag_amp is None else max_wavelet_mag_amp
        )
        self.max_wavelet_phase_amp = (
            max_wavelet_amp if max_wavelet_phase_amp is None else max_wavelet_phase_amp
        )
        self.wavelet_level_amp_scales = parse_wavelet_level_amp_scales(
            wavelet_level_amp_scales, wavelet_J
        )
        self.wavelet_polar_mode = wavelet_polar_mode
        self.wavelet_per_channel = wavelet_per_channel
        self.wavelet_spatial_grid = int(wavelet_spatial_grid)
        self.wavelet_yl_gain = wavelet_yl_gain
        self.max_wavelet_yl_amp = max_wavelet_yl_amp

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
        self.mask_shape = mask_shape

        self.disable_dct = disable_dct
        self.disable_flow = disable_flow
        self.disable_photo = disable_photo

        # ----- Parâmetros DCT (sempre existem, mas só usados se transform_type == 'dct')
        self.raw_dct = nn.Parameter(torch.zeros(1, 3, self.dct_k, self.dct_k))

        # ----- Parâmetros DT-CWT (somente se transform_type == 'dtcwt') -----
        if transform_type == "dtcwt":
            from .wavelet_transform import DTCWTDecimated

            self.wavelet = DTCWTDecimated(J=wavelet_J)
            g = self.wavelet_spatial_grid
            if g > 0:
                # [1, 6, J, g, g] — mapas espaciais de baixa resolução
                self.raw_wavelet_mag_scale = nn.Parameter(
                    torch.zeros(1, 6, wavelet_J, g, g)
                )
                self.raw_wavelet_phase_shift = nn.Parameter(
                    torch.zeros(1, 6, wavelet_J, g, g)
                )
            elif wavelet_per_channel:
                # [1, 3, 6, J]
                self.raw_wavelet_mag_scale = nn.Parameter(
                    torch.zeros(1, 3, 6, wavelet_J)
                )
                self.raw_wavelet_phase_shift = nn.Parameter(
                    torch.zeros(1, 3, 6, wavelet_J)
                )
            else:
                # [1, 6, J] — global por orientação × nível
                self.raw_wavelet_mag_scale = nn.Parameter(torch.zeros(1, 6, wavelet_J))
                self.raw_wavelet_phase_shift = nn.Parameter(
                    torch.zeros(1, 6, wavelet_J)
                )

            if wavelet_yl_gain:
                self.raw_wavelet_yl_gain = nn.Parameter(torch.zeros(1, 1, 1, 1))
            else:
                self.raw_wavelet_yl_gain = None
        else:
            self.wavelet = None
            self.raw_wavelet_mag_scale = None
            self.raw_wavelet_phase_shift = None
            self.raw_wavelet_yl_gain = None

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

    def _wavelet_level_modulators(
        self, level: int, spatial_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scale, phase_shift) broadcastable to complex coeffs [B,C,6,H,W]."""
        h, w = spatial_hw
        level_scale = self.wavelet_level_amp_scales[level]
        mag_amp = self.max_wavelet_mag_amp * level_scale
        phase_amp = self.max_wavelet_phase_amp * level_scale

        if self.wavelet_spatial_grid > 0:
            raw_mag = self.raw_wavelet_mag_scale[:, :, level]  # [1, 6, g, g]
            raw_ph = self.raw_wavelet_phase_shift[:, :, level]
            raw_mag = F.interpolate(
                raw_mag, size=(h, w), mode="bicubic", align_corners=False
            )
            raw_ph = F.interpolate(
                raw_ph, size=(h, w), mode="bicubic", align_corners=False
            )
            # [1, 6, H, W] -> [1, 1, 6, H, W]
            scale = 1.0 + mag_amp * torch.tanh(raw_mag).unsqueeze(1)
            phase_shift = phase_amp * torch.tanh(raw_ph).unsqueeze(1)
        elif self.wavelet_per_channel:
            raw_mag = self.raw_wavelet_mag_scale[:, :, :, level]  # [1, 3, 6]
            raw_ph = self.raw_wavelet_phase_shift[:, :, :, level]
            scale = 1.0 + mag_amp * torch.tanh(raw_mag).unsqueeze(-1).unsqueeze(-1)
            phase_shift = phase_amp * torch.tanh(raw_ph).unsqueeze(-1).unsqueeze(-1)
        else:
            raw_mag = self.raw_wavelet_mag_scale[:, :, level]  # [1, 6]
            raw_ph = self.raw_wavelet_phase_shift[:, :, level]
            scale = 1.0 + mag_amp * torch.tanh(raw_mag).view(1, 1, 6, 1, 1)
            phase_shift = phase_amp * torch.tanh(raw_ph).view(1, 1, 6, 1, 1)

        return scale, phase_shift

    def _perturb_wavelet_coeffs_legacy(self, coeffs: torch.Tensor, level: int) -> torch.Tensor:
        """Historical element-wise abs/angle on real/imag stack [B,C,6,H,W,2]."""
        mag = torch.abs(coeffs)
        phase = torch.angle(coeffs)
        level_scale = self.wavelet_level_amp_scales[level]
        mag_amp = self.max_wavelet_mag_amp * level_scale
        phase_amp = self.max_wavelet_phase_amp * level_scale
        scale = 1.0 + mag_amp * torch.tanh(
            self.raw_wavelet_mag_scale[:, :, level].view(1, -1, 1, 1, 1)
        )
        phase_shift = phase_amp * torch.tanh(
            self.raw_wavelet_phase_shift[:, :, level].view(1, -1, 1, 1, 1)
        )
        new_mag = mag * scale
        new_phase = phase + phase_shift
        return new_mag * torch.exp(1j * new_phase)

    def _perturb_wavelet_coeffs_complex(self, coeffs: torch.Tensor, level: int) -> torch.Tensor:
        """Polar mag/phase on true complex coeffs; returns real/imag stack for inverse."""
        # pytorch-wavelets: [B, C, 6, H, W, 2]
        c = torch.view_as_complex(coeffs.contiguous())
        mag = torch.abs(c)
        phase = torch.angle(c)
        scale, phase_shift = self._wavelet_level_modulators(level, (c.shape[-2], c.shape[-1]))
        new_c = (mag * scale) * torch.exp(1j * (phase + phase_shift))
        return torch.view_as_real(new_c)

    def forward(self, x: torch.Tensor, face_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        device = x.device

        if face_mask is None:
            face_mask = self.get_face_mask(b, device)

        y = x

        # Transformada espectral (DCT ou Wavelet)
        if not self.disable_dct:
            if self.transform_type == "dct":
                dct_delta = self.reconstruct_dct_delta(b, device, face_mask)
                y = y + dct_delta
                y = y.clamp(0.0, 1.0)
            elif self.transform_type == "dtcwt":
                Yl, Yh = self.wavelet(x)
                if self.wavelet_yl_gain and self.raw_wavelet_yl_gain is not None:
                    yl_scale = 1.0 + self.max_wavelet_yl_amp * torch.tanh(
                        self.raw_wavelet_yl_gain
                    )
                    Yl = Yl * yl_scale

                Yh_perturbed = []
                for level, coeffs in enumerate(Yh):
                    # coeffs: [B, C, 6, H_l, W_l, 2] (real/imag)
                    if self.wavelet_polar_mode == "legacy":
                        new_coeffs = self._perturb_wavelet_coeffs_legacy(coeffs, level)
                    else:
                        new_coeffs = self._perturb_wavelet_coeffs_complex(coeffs, level)
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

        if self.transform_type == "dct":
            reg["dct_l2"] = (torch.tanh(self.raw_dct) ** 2).mean()
        elif self.transform_type == "dtcwt" and self.raw_wavelet_mag_scale is not None:
            mag_l2 = (torch.tanh(self.raw_wavelet_mag_scale) ** 2).mean()
            phase_l2 = (torch.tanh(self.raw_wavelet_phase_shift) ** 2).mean()

            reg["wavelet_mag_l2"] = mag_l2
            reg["wavelet_phase_l2"] = phase_l2

            raw = self.raw_wavelet_mag_scale
            if raw.dim() == 3:
                smooth_mag = (raw[:, :, 1:] - raw[:, :, :-1]).abs().mean()
            elif raw.dim() == 4:
                smooth_mag = (raw[:, :, :, 1:] - raw[:, :, :, :-1]).abs().mean()
            elif raw.dim() == 5:
                smooth_mag = (raw[:, :, 1:] - raw[:, :, :-1]).abs().mean()
            else:
                smooth_mag = raw.new_zeros(())

            reg["wavelet_mag_smooth"] = smooth_mag

            if self.raw_wavelet_yl_gain is not None:
                reg["wavelet_yl_l2"] = (torch.tanh(self.raw_wavelet_yl_gain) ** 2).mean()

        return reg
