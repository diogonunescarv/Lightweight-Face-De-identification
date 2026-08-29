import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from models.transformer import LightweightDeIdentifier
from models.embedders import load_authorized_embedders
from models.masks import resolve_face_mask_for_model
from utils.checkpoint import load_transformer_from_checkpoint
from data.dataset import FaceImageFolder
from losses.losses import ssim_index_masked

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)

    # Carrega o transformador
    transformer = load_transformer_from_checkpoint(args.checkpoint, device)
    transformer.eval()

    mask_mode = getattr(args, "mask_mode", None)
    mask_regions = getattr(args, "mask_regions", None)
    landmark_detector = None
    effective_mode = mask_mode or transformer.mask_mode
    if effective_mode == "landmarks" and transformer.use_face_mask:
        from models.face_detector import get_default_detector

        landmark_detector = get_default_detector()

    # Carrega os embedders (os mesmos usados no treino)
    embedders = load_authorized_embedders(device)

    # Dataset de validação (não o mesmo do treino)
    dataset = FaceImageFolder(args.data, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_cos = []
    all_ssim = []
    total_images = 0

    for x, paths in loader:
        x = x.to(device)

        face_mask = resolve_face_mask_for_model(
            transformer,
            paths,
            device,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            detector=landmark_detector,
        )
        y = transformer(x, face_mask=face_mask)

        # Similaridade cosseno (média sobre todos os embedders)
        batch_cos = []
        for model in embedders:
            e0 = F.normalize(model(x), dim=1)
            e1 = F.normalize(model(y), dim=1)
            cos = (e0 * e1).sum(dim=1)      # shape [B]
            batch_cos.append(cos)
        # Média sobre os embedders (se mais de um)
        mean_cos = torch.stack(batch_cos).mean(dim=0)   # [B]
        all_cos.append(mean_cos)

        # SSIM SEM MASCARA
        #ssim_vals = ssim_index(x, y)   # [B]

        face_mask = resolve_face_mask_for_model(
            transformer,
            paths,
            device,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            detector=landmark_detector,
        )
        ssim_vals = ssim_index_masked(x, y, face_mask)   # [B]
        all_ssim.append(ssim_vals)

        total_images += x.shape[0]

    # Concatena os tensores de todos os batches
    cos_all = torch.cat(all_cos)        # [N]
    ssim_all = torch.cat(all_ssim)      # [N]

    mean_cos = cos_all.mean().item()
    mean_ssim = ssim_all.mean().item()
    # Distância euclidiana média no espaço de embeddings
    mean_euclid = math.sqrt(2.0 * (1.0 - mean_cos))

    print("\n========== Evaluation Results ==========")
    print(f"Total images evaluated: {total_images}")
    print(f"Mean cosine similarity: {mean_cos:.4f}")
    print(f"Mean Euclidean distance: {mean_euclid:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print("========================================\n")

    # Opcional: salvar os resultados em um arquivo
    if args.output_summary:
        out_path = Path(args.output_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(f"images,{total_images}\n")
            f.write(f"cos_mean,{mean_cos:.6f}\n")
            f.write(f"euclid_mean,{mean_euclid:.6f}\n")
            f.write(f"ssim_mean,{mean_ssim:.6f}\n")
        print(f"Summary saved to {out_path}")