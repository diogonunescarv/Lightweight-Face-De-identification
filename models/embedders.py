import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
def load_authorized_embedders(device):
    """Retorna lista de modelos (apenas FaceNet por enquanto)."""
    model = FaceNetEmbedder(pretrained='vggface2')
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return [model]

def normalized_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    emb = model(x)
    emb = F.normalize(emb, dim=1)
    return emb

# Código antigo (apenas como referencia por enquanto)

'''
def build_face_embedders(device):
    model = FaceNetEmbedder(pretrained='vggface2')
    model.to(device)
    model.eval()
    # Certifique-se de que os gradientes estão desligados
    for p in model.parameters():
        p.requires_grad_(False)
    return [model]

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
    
'''