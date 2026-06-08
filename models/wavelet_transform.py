import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class DTCWTUndecimated(nn.Module):
    def __init__(self, J=3, biort='near_sym_a', qshift='qshift_a'):
        super().__init__()
        self.J = J
        self.forward = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.inverse = DTCWTInverse(biort=biort, qshift=qshift)

    def forward(self, x):
        # Retorna (Yl, Yh) onde Yh é lista de tensores para cada nível
        return self.forward(x)

    def inverse(self, Yl, Yh):
        return self.inverse((Yl, Yh))