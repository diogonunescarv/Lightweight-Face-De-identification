"""
Avaliação de um único modelo, salvando métricas (JSON/CSV) e
opcionalmente as imagens transformadas para inspeção visual.
Inclui salvamento de resultados por imagem (CSV individual).
"""

import json
import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import FaceImageFolder, tensor_to_pil
from losses.losses import ssim_index_masked, identity_loss_ensemble
from models.embedders import load_authorized_embedders
from models.masks import resolve_face_mask_for_model
from utils.checkpoint import load_transformer_from_checkpoint


@torch.no_grad()
def evaluate_single_model(
    checkpoint_path: str,
    data_dir: str,
    output_file: str,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int = 0,
    save_images: bool = False,
    max_visual_samples: int = 10,
    mask_mode: str | None = None,
    mask_regions: str | None = None,
) -> dict:
    device = torch.device(device)
    out_base = Path(output_file)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    
    model = load_transformer_from_checkpoint(checkpoint_path, device)
    embedders = load_authorized_embedders(device)

    landmark_detector = None
    effective_mode = mask_mode or model.mask_mode
    if effective_mode == "landmarks" and model.use_face_mask:
        from models.face_detector import get_default_detector

        landmark_detector = get_default_detector()
    
    dataset = FaceImageFolder(data_dir, model.image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    
    all_cos = []
    all_euclid = []
    all_ssim = []
    all_times_ms = []
    per_image_data = []  # lista de dicionários para cada imagem
    total_images = 0
    saved_visual = 0
    
    if save_images:
        img_dir = out_base.parent / f"{out_base.stem}_transformed"
        img_dir.mkdir(parents=True, exist_ok=True)
        print(f"Salvando imagens transformadas em: {img_dir}")
    else:
        img_dir = None
    
    print(f"Avaliando modelo: {checkpoint_path}")
    for x, paths in loader:
        if max_samples > 0 and total_images >= max_samples:
            break
        x = x.to(device)

        face_mask = resolve_face_mask_for_model(
            model,
            paths,
            device,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            detector=landmark_detector,
        )
        
        # Medir tempo
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x, face_mask=face_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_time_ms = (t1 - t0) * 1000 / x.size(0)
        
        # Cosine similarity por imagem (média sobre embedders)
        batch_cos = []
        for emb in embedders:
            e0 = F.normalize(emb(x), dim=1)
            e1 = F.normalize(emb(y), dim=1)
            cos_batch = (e0 * e1).sum(dim=1)
            batch_cos.append(cos_batch)
        mean_cos_batch = torch.stack(batch_cos).mean(dim=0)   # [B]
        euclid_batch = torch.sqrt(2.0 * (1.0 - mean_cos_batch))
        
        # SSIM com máscara
        ssim_vals = ssim_index_masked(x, y, face_mask)  # [B]
        
        # Armazenar resultados por imagem
        for i in range(x.size(0)):
            per_image_data.append({
                "image": Path(paths[i]).name,
                "cosine": mean_cos_batch[i].item(),
                "euclidean": euclid_batch[i].item(),
                "ssim": ssim_vals[i].item(),
                "inference_time_ms": batch_time_ms,
            })
        
        all_cos.extend(mean_cos_batch.cpu().tolist())
        all_euclid.extend(euclid_batch.cpu().tolist())
        all_ssim.extend(ssim_vals.cpu().tolist())
        all_times_ms.extend([batch_time_ms] * x.size(0))
        
        # Salvar imagens visuais
        if save_images and saved_visual < max_visual_samples:
            for i in range(min(x.size(0), max_visual_samples - saved_visual)):
                orig_pil = tensor_to_pil(x[i])
                trans_pil = tensor_to_pil(y[i])
                src_name = Path(paths[i]).stem
                trans_pil.save(img_dir / f"{src_name}_transformed.jpg", quality=95)
                # Comparação lado a lado
                w, h = orig_pil.size
                grid = Image.new('RGB', (w*2, h), (255,255,255))
                grid.paste(orig_pil, (0,0))
                grid.paste(trans_pil, (w,0))
                grid.save(img_dir / f"{src_name}_compare.jpg")
                saved_visual += 1
            print(f"Salvas {saved_visual}/{max_visual_samples} imagens visuais", end='\r')
        
        total_images += x.size(0)
        print(f"Processadas {total_images} imagens", end='\r')
    
    print()
    
    # Salvar resultados por imagem em CSV
    per_image_csv = out_base.with_suffix(".per_image.csv")
    with open(per_image_csv, "w", newline="") as f:
        fieldnames = ["image", "cosine", "euclidean", "ssim", "inference_time_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_data)
    print(f"Resultados por imagem salvos em: {per_image_csv}")
    
    # Métricas consolidadas
    metrics = {
        "model": str(Path(checkpoint_path).name),
        "checkpoint": checkpoint_path,
        "n_images": total_images,
        "cos_mean": float(torch.tensor(all_cos).mean()),
        "cos_std": float(torch.tensor(all_cos).std()),
        "euclid_mean": float(torch.tensor(all_euclid).mean()),
        "euclid_std": float(torch.tensor(all_euclid).std()),
        "ssim_mean": float(torch.tensor(all_ssim).mean()),
        "ssim_std": float(torch.tensor(all_ssim).std()),
        "inference_time_ms_mean": float(torch.tensor(all_times_ms).mean()),
        "inference_time_ms_std": float(torch.tensor(all_times_ms).std()),
    }
    
    # Salvar JSON e CSV (resumo)
    json_path = out_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    
    print(f"Métricas resumidas salvas em: {json_path} e {csv_path}")
    
    print("\n" + "="*50)
    print(f"Resultados para {metrics['model']}")
    print(f"  Cosine similarity: {metrics['cos_mean']:.4f} ± {metrics['cos_std']:.4f}")
    print(f"  Euclidean distance: {metrics['euclid_mean']:.4f} ± {metrics['euclid_std']:.4f}")
    print(f"  SSIM (masked): {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"  Inference time (ms/img): {metrics['inference_time_ms_mean']:.2f} ± {metrics['inference_time_ms_std']:.2f}")
    print("="*50 + "\n")
    
    return metrics