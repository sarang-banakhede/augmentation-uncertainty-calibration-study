"""
evaluate.py — Evaluate trained models on PH2 dataset (out-of-distribution).

Usage:
    python evaluate.py
    python evaluate.py --weights_root /path/to/results --ph2_img /path/img --ph2_lbl /path/lbl
"""
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config as cfg
from src.model import UNet
from src.datasets import PH2Dataset
from src.engine import mc_inference
from src.metrics import (
    compute_ece, compute_brier, compute_hd95,
    compute_predictive_entropy, per_image_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_root", default=str(cfg.RESULTS_ROOT))
    p.add_argument("--ph2_img",      default=str(cfg.PH2_IMAGE_DIR))
    p.add_argument("--ph2_lbl",      default=str(cfg.PH2_LABEL_DIR))
    p.add_argument("--mc_passes",    type=int, default=cfg.MC_PASSES)
    p.add_argument("--strategies",   nargs="+", default=cfg.STRATEGIES)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PH2Dataset(args.ph2_img, args.ph2_lbl, cfg.IMAGE_SIZE)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True)
    print(f"PH2 dataset: {len(dataset)} images\n")

    for strategy in args.strategies:
        print(f"{'=' * 60}\n  STRATEGY: {strategy.upper()}\n{'=' * 60}")

        weights_path = Path(args.weights_root) / strategy / "best_weights.pth"
        if not weights_path.exists():
            print(f"  [SKIP] Weights not found: {weights_path}")
            continue

        model = UNet(in_ch=3, out_ch=1, dropout_p=cfg.DROPOUT_P).to(device)
        state = torch.load(weights_path, map_location=device)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        del state

        rows, all_probs, all_labels = [], [], []

        for imgs, masks, names in tqdm(loader, desc=f"[{strategy}]"):
            mean_pred, var_map = mc_inference(model, imgs, args.mc_passes, device)
            mean_np = mean_pred[0, 0]
            var_np  = var_map[0, 0]
            gt_np   = masks[0, 0].numpy()

            seg  = per_image_metrics(mean_np, gt_np)
            hd95 = compute_hd95(mean_np, gt_np)
            unc  = float(var_np.mean())
            ent  = compute_predictive_entropy(mean_np)

            all_probs.append(mean_np.flatten())
            all_labels.append(gt_np.flatten())

            rows.append({
                "image":           names[0],
                "dice":            round(seg["dice"],      4),
                "iou":             round(seg["iou"],       4),
                "precision":       round(seg["precision"], 4),
                "recall":          round(seg["recall"],    4),
                "accuracy":        round(seg["accuracy"],  4),
                "hd95":            round(hd95, 4) if not np.isnan(hd95) else None,
                "avg_uncertainty": round(unc, 6),
                "pred_entropy":    round(ent, 6),
            })

        ece   = compute_ece(all_probs, all_labels, cfg.ECE_BINS)
        brier = compute_brier(all_probs, all_labels)
        for row in rows:
            row["ece"]         = round(ece, 6)
            row["brier_score"] = round(brier, 6)

        df       = pd.DataFrame(rows)
        out_path = Path(args.weights_root) / strategy / "ph2_inference.csv"
        df.to_csv(out_path, index=False)

        print(f"  Dice: {df['dice'].mean():.4f} | IoU: {df['iou'].mean():.4f} | "
              f"HD95: {df['hd95'].mean():.4f} | ECE: {ece:.4f} | Brier: {brier:.4f}")
        print(f"  Saved → {out_path}")

        del model; torch.cuda.empty_cache(); gc.collect()

    print("\nPH2 evaluation complete.")


if __name__ == "__main__":
    main()
