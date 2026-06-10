"""
visualize.py — MC-Dropout uncertainty visualization for a single image across all strategies.

Usage:
    python visualize.py --image /path/to/image.png
    python visualize.py --image /path/to/image.png --weights_root results/ --save_dir visualizations/
"""
import gc
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.model import UNet
from src.metrics import compute_predictive_entropy

BG     = "#F5F0E8"
DARK   = "#2C2C2A"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",        required=True,             help="Path to input image")
    p.add_argument("--weights_root", default=str(cfg.RESULTS_ROOT))
    p.add_argument("--save_dir",     default=str(cfg.VIZ_DIR))
    p.add_argument("--mc_passes",    type=int, default=cfg.MC_PASSES)
    p.add_argument("--strategies",   nargs="+", default=cfg.STRATEGIES)
    return p.parse_args()


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = UNet(in_ch=3, out_ch=1, dropout_p=cfg.DROPOUT_P).to(device)
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def preprocess(img_path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    img  = np.array(Image.open(img_path).convert("RGB"))
    tf   = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
        ToTensorV2(),
    ])
    orig = A.Resize(image_size, image_size)(image=img)["image"]
    return tf(image=img)["image"].float().unsqueeze(0), orig


def mc_forward(model: nn.Module, img_tensor: torch.Tensor, n_passes: int, device: torch.device):
    from src.engine import enable_dropout
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(img_tensor.to(device)).cpu().numpy())
    preds = np.stack(preds, axis=0)        # [T, 1, 1, H, W]
    return preds.mean(axis=0)[0, 0], preds.var(axis=0)[0, 0]


def _style_axes(axes, color):
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(2)


def save_per_strategy(orig, mean_map, var_map, strategy, label, color, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG)
    axes[0].imshow(orig)
    im1 = axes[1].imshow(mean_map, cmap="RdYlGn", vmin=0, vmax=1)
    im2 = axes[2].imshow(var_map,  cmap="hot",    vmin=0, vmax=var_map.max())
    for ax, im, title in zip(axes,
                              [None, im1, im2],
                              ["Input Image", "Mean Prediction (p̄)", "Predictive Variance (σ²)"]):
        ax.set_title(title, fontsize=10, fontweight="bold", color=DARK)
        if im:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_axes(axes, color)
    h = compute_predictive_entropy(mean_map)
    fig.suptitle(f"{label} | σ²={var_map.mean():.6f} | H={h:.4f}",
                 fontsize=11, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / f"{strategy}_mc_uncertainty.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_combined(orig, results, save_dir, mc_passes):
    fig, axes = plt.subplots(5, 3, figsize=(13, 18), facecolor=BG)
    for col, title in enumerate(["Input", "Mean Prediction (p̄)", "Predictive Variance (σ²)"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold", color=DARK, pad=8)

    for row, (_, label, color, mean_map, var_map) in enumerate(results):
        axes[row][0].imshow(orig)
        axes[row][0].set_ylabel(label, fontsize=10, fontweight="bold", color=color, rotation=90, labelpad=10)
        im1 = axes[row][1].imshow(mean_map, cmap="RdYlGn", vmin=0, vmax=1)
        im2 = axes[row][2].imshow(var_map,  cmap="hot",    vmin=0, vmax=var_map.max())
        plt.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)
        h = compute_predictive_entropy(mean_map)
        axes[row][2].set_xlabel(f"σ²={var_map.mean():.5f} | H={h:.4f}", fontsize=8, color=DARK)
        _style_axes(axes[row], color)

    fig.suptitle(f"MC-Dropout Uncertainty  (T={mc_passes} passes)",
                 fontsize=13, fontweight="bold", color=DARK, y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "combined_all_strategies.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_variance_comparison(results, save_dir, mc_passes):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG)
    vmax = max(var_map.max() for *_, _, var_map in results)
    for i, (_, label, color, mean_map, var_map) in enumerate(results):
        im = axes[i].imshow(var_map, cmap="hot", vmin=0, vmax=vmax)
        axes[i].set_title(label, fontsize=10, fontweight="bold", color=color)
        h = compute_predictive_entropy(mean_map)
        axes[i].set_xlabel(f"σ²={var_map.mean():.5f}\nH={h:.4f}", fontsize=8.5, color=DARK)
        _style_axes(axes[i], color)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    fig.suptitle(f"Predictive Variance σ²  (T={mc_passes} passes, shared colorscale)",
                 fontsize=11, fontweight="bold", color=DARK)
    fig.tight_layout()
    fig.savefig(save_dir / "variance_comparison.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_mean_comparison(orig, results, save_dir, mc_passes):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4), facecolor=BG)
    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=10, fontweight="bold", color=DARK)
    axes[0].axis("off")
    for i, r in enumerate(results):
        label, color, mean_map = r[1], r[2], r[3]
        axes[i + 1].imshow(mean_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i + 1].set_title(label, fontsize=10, fontweight="bold", color=color)
        fg = (mean_map > 0.5).mean() * 100
        axes[i + 1].set_xlabel(f"Foreground: {fg:.1f}%", fontsize=8.5, color=DARK)
        _style_axes(axes[i + 1], color)
    fig.suptitle(f"Mean Prediction Comparison  (T={mc_passes} passes)",
                 fontsize=11, fontweight="bold", color=DARK)
    fig.tight_layout()
    fig.savefig(save_dir / "mean_prediction_comparison.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    args     = parse_args()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image   : {args.image}")
    print(f"Weights : {args.weights_root}")
    print(f"Output  : {args.save_dir}")
    print(f"Passes  : {args.mc_passes}\n")

    img_tensor, orig = preprocess(args.image, cfg.IMAGE_SIZE)
    labels  = dict(zip(cfg.STRATEGIES, cfg.LABELS))
    colors  = dict(zip(cfg.STRATEGIES, cfg.COLORS))
    results = []

    for strategy in args.strategies:
        weights_path = Path(args.weights_root) / strategy / "best_weights.pth"
        if not weights_path.exists():
            print(f"  [SKIP] {strategy}: weights not found")
            continue

        print(f"  [{labels[strategy]}] running {args.mc_passes} MC passes...")
        model    = load_model(weights_path, device)
        mean_map, var_map = mc_forward(model, img_tensor, args.mc_passes, device)

        save_per_strategy(orig, mean_map, var_map, strategy,
                          labels[strategy], colors[strategy], save_dir)
        np.save(save_dir / f"{strategy}_mean.npy",     mean_map)
        np.save(save_dir / f"{strategy}_variance.npy", var_map)

        results.append((strategy, labels[strategy], colors[strategy], mean_map, var_map))
        del model; torch.cuda.empty_cache(); gc.collect()

    if results:
        print("\nSaving summary figures...")
        save_combined(orig, results, save_dir, args.mc_passes)
        save_variance_comparison(results, save_dir, args.mc_passes)
        save_mean_comparison(orig, results, save_dir, args.mc_passes)

    print(f"\nDone. Outputs saved to: {save_dir}/")


if __name__ == "__main__":
    main()
