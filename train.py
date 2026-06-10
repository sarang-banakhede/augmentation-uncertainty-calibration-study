"""
train.py — Train U-Net with 5 augmentation strategies on ISIC-2016 and run MC-Dropout inference.

Usage:
    python train.py
    python train.py --data_root /path/to/ISIC2016 --results_root /path/to/results
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
from src.datasets import SkinDataset
from src.augmentations import get_transforms
from src.engine import train_one_epoch, eval_one_epoch, mc_inference
from src.metrics import compute_ece, compute_brier, compute_hd95, compute_predictive_entropy, per_image_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default=str(cfg.ISIC_DATA_ROOT))
    p.add_argument("--results_root", default=str(cfg.RESULTS_ROOT))
    p.add_argument("--epochs",       type=int,   default=cfg.NUM_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=cfg.LR)
    p.add_argument("--mc_passes",    type=int,   default=cfg.MC_PASSES)
    p.add_argument("--strategies",   nargs="+",  default=cfg.STRATEGIES)
    return p.parse_args()


def run_isic_inference(model, loader, mc_passes, ece_bins, device):
    rows, all_probs, all_labels = [], [], []

    for imgs, masks, names in loader:
        mean_pred, var_map = mc_inference(model, imgs, mc_passes, device)
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

    ece   = compute_ece(all_probs, all_labels, ece_bins)
    brier = compute_brier(all_probs, all_labels)
    for row in rows:
        row["ece"]         = round(ece, 6)
        row["brier_score"] = round(brier, 6)

    return pd.DataFrame(rows), ece, brier


def main():
    args = parse_args()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu   = torch.cuda.device_count() > 1
    criterion   = nn.BCELoss()
    data_root   = Path(args.data_root)
    results_root = Path(args.results_root)

    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    for strategy in args.strategies:
        print(f"\n{'=' * 60}\n  STRATEGY: {strategy.upper()}\n{'=' * 60}")

        out_dir = results_root / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        train_tf, test_tf = get_transforms(strategy, cfg.IMAGE_SIZE)

        train_ds = SkinDataset(data_root / "train" / "images", data_root / "train" / "masks", train_tf)
        test_ds  = SkinDataset(data_root / "test"  / "images", data_root / "test"  / "masks", test_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)

        torch.cuda.empty_cache(); gc.collect()

        model = UNet(in_ch=3, out_ch=1, dropout_p=cfg.DROPOUT_P)
        if multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_dice = 0.0
        history   = []

        for epoch in tqdm(range(1, args.epochs + 1), desc=f"[{strategy}]", unit="epoch"):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, strategy, device)
            te = eval_one_epoch(model, test_loader, criterion, device)
            scheduler.step()

            row = {"epoch": epoch}
            row.update({f"train_{k}": round(v, 4) for k, v in tr.items()})
            row.update({f"test_{k}":  round(v, 4) for k, v in te.items()})
            history.append(row)

            torch.save(model.state_dict(), out_dir / "last_weights.pth")
            if te["dice"] > best_dice:
                best_dice = te["dice"]
                torch.save(model.state_dict(), out_dir / "best_weights.pth")

            tqdm.write(
                f"  Ep {epoch:3d}/{args.epochs} | "
                f"Tr loss:{tr['loss']:.4f} dice:{tr['dice']:.4f} | "
                f"Te loss:{te['loss']:.4f} dice:{te['dice']:.4f}"
                + (" ★" if te["dice"] == best_dice else "")
            )

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        print(f"  Best test Dice: {best_dice:.4f}")

        del optimizer, scheduler
        torch.cuda.empty_cache(); gc.collect()

        # Reload best weights for inference
        del model
        torch.cuda.empty_cache(); gc.collect()
        model = UNet(in_ch=3, out_ch=1, dropout_p=cfg.DROPOUT_P)
        if multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)
        state = torch.load(out_dir / "best_weights.pth", map_location=device)
        model.load_state_dict(state)
        del state; torch.cuda.empty_cache()

        print(f"  Running MC inference ({args.mc_passes} passes) on ISIC test set...")
        df, ece, brier = run_isic_inference(model, test_loader, args.mc_passes, cfg.ECE_BINS, device)
        df.to_csv(out_dir / "isic_inference.csv", index=False)
        print(f"  ECE: {ece:.4f} | Brier: {brier:.4f}")
        print(f"  Saved → {out_dir}/")

        del model, train_loader, test_loader, train_ds, test_ds, df
        torch.cuda.empty_cache(); gc.collect()

    print("\nAll experiments complete.")
    print(f"Results saved to: {results_root}/")
    for s in args.strategies:
        print(f"  {s}/  training_history.csv | isic_inference.csv | best_weights.pth")


if __name__ == "__main__":
    main()
