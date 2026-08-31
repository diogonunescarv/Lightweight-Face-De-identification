import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets.dtcwt import lowlevel as _dtcwt_lowlevel
from pytorch_wavelets.dtcwt import transform_funcs as _dtcwt_transform_funcs


def _patch_complex_filters(fn):
    def wrapped(X, h, *args, **kwargs):
        if torch.is_tensor(X) and torch.is_tensor(h):
            if torch.is_complex(X) and not torch.is_complex(h):
                h = h.to(X.dtype)
        return fn(X, h, *args, **kwargs)
    return wrapped


_FILTER_FN_NAMES = (
    "colifilt", "rowifilt", "colfilter", "rowfilter", "coldfilt", "rowdfilt",
)
for _module in (_dtcwt_lowlevel, _dtcwt_transform_funcs):
    for _name in _FILTER_FN_NAMES:
        if hasattr(_module, _name):
            setattr(_module, _name, _patch_complex_filters(getattr(_module, _name)))


class DTCWTUndecimated(nn.Module):
    def __init__(self, J=3, biort='near_sym_a', qshift='qshift_a'):
        super().__init__()
        self.J = J
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)

    def forward(self, x):
        return self.xfm(x)

    def inverse(self, Yl, Yh):
        y = self.ifm((Yl, Yh))
        if torch.is_complex(y):
            y = y.real
        return y