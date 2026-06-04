import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.transformer import LightweightDeIdentifier
from data.dataset import FaceImageFolder, tensor_to_pil
from utils.checkpoint import load_transformer_from_checkpoint

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