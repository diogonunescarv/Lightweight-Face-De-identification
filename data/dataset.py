from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

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