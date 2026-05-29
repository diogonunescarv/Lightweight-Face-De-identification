from __future__ import annotations

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetEmbedder(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super().__init__()
        self.model = InceptionResnetV1(pretrained=pretrained).eval()
        # Congela os pesos
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: [B,3,H,W] em [0,1]
        # O FaceNet espera entrada normalizada entre -1 e 1? Depende.
        # O modelo do facenet-pytorch espera [0,1] e faz a normalização interna.
        # Portanto, basta passar x.
        return self.model(x)


def build_face_embedders(device):
    model = FaceNetEmbedder(pretrained='vggface2')
    model.to(device)
    model.eval()
    # Certifique-se de que os gradientes estão desligados
    for p in model.parameters():
        p.requires_grad_(False)
    return [model]