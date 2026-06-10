"""
plot_results.py — Generate all analysis figures from ISIC and PH2 inference results.

Usage:
    python plot_results.py
    python plot_results.py --results_root results/ --figures_dir figures/
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

from src import config as cfg

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.linewidth":    1.3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
})

BG     = "white"
DARK   = "#1A1A1A"
ACCENT = "#555555"
GRID_C = "#EBEBEB"
HATCH  = "////"
DPI    = 400
W, H   = 9, 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(cfg.RESULTS_ROOT))
    p.add_argument("--figures_dir",  default=str(cfg.FIGURES_DIR))
    return p.parse_args()


def load_data(results_root: str) -> tuple[dict, dict]:
    isic, ph2 = {}, {}
    for s in cfg.STRATEGIES:
        isic[s] = pd.read_csv(Path(results_root) / s / "isic_inference.csv")
        ph2[s]  = pd.read_csv(Path(results_root) / s / "ph2_inference.csv")
    return isic, ph2


def _style(ax, xlabel="", ylabel="", fs=11):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DARK, labelsize=fs - 1, length=5, width=1.1)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#BBBBBB")
        sp.set_linewidth(1.2)
    ax.grid(True, color=GRID_C, linewidth=0.8, linestyle="--", alpha=0.9, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs, labelpad=5)


def _bar_labels(ax, xs, vals, pad_frac=0.022, fmt=".3f", fs=9):
    pad = max(vals) * pad_frac
    for x, v in zip(xs, vals):
        ax.text(x, v + pad, f"{v:{fmt}}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color=DARK)


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. DSC bar chart ──────────────────────────────────────────────────────────
def plot_dsc(isic, ph2, out_dir):
    dice_i = [isic[s].dice.mean() for s in cfg.STRATEGIES]
    dice_p = [ph2[s].dice.mean()  for s in cfg.STRATEGIES]
    x, w   = np.arange(5), 0.28
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    ax.bar(x - w/2, dice_i, w, color=[c + "DD" for c in cfg.COLORS],
           edgecolor=DARK, lw=1.0, label="ISIC-2016", zorder=3)
    ax.bar(x + w/2, dice_p, w, color=cfg.COLORS, edgecolor=DARK, lw=1.0,
           label="PH2 (OOD)", hatch=HATCH, zorder=3)
    _bar_labels(ax, x - w/2, dice_i)
    _bar_labels(ax, x + w/2, dice_p)
    ax.set_xticks(x); ax.set_xticklabels(cfg.LABELS, fontsize=10)
    ax.set_ylim(0, max(max(dice_i), max(dice_p)) * 1.28)
    ax.legend(fontsize=10, framealpha=1, edgecolor="#BBBBBB", loc="upper right")
    _style(ax, "Augmentation Strategy", "Mean DSC (↑ better)")
    _save(fig, out_dir, "dsc_bars.png")


# ── 2. ECE bar chart with Δ ───────────────────────────────────────────────────
def plot_ece(isic, ph2, out_dir):
    ece_i = [isic[s].ece.mean() for s in cfg.STRATEGIES]
    ece_p = [ph2[s].ece.mean()  for s in cfg.STRATEGIES]
    x, w  = np.arange(5), 0.28
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    ax.bar(x - w/2, ece_i, w, color=[c + "DD" for c in cfg.COLORS],
           edgecolor=DARK, lw=1.0, label="ISIC-2016", zorder=3)
    ax.bar(x + w/2, ece_p, w, color=cfg.COLORS, edgecolor=DARK, lw=1.0,
           label="PH2 (OOD)", hatch=HATCH, zorder=3)
    pad = max(max(ece_i), max(ece_p)) * 0.022
    for i, (a, b) in enumerate(zip(ece_i, ece_p)):
        ax.text(x[i] - w/2, a + pad, f"{a:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=DARK)
        ax.text(x[i] + w/2, b + pad, f"{b:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=DARK)
        delta = b - a
        ax.text(x[i], max(a, b) + pad * 4.5, f"Δ{delta:+.3f}", ha="center",
                fontsize=9.5, fontweight="bold",
                color="#C0392B" if delta > 0 else "#1A7A2A")
    ax.set_xticks(x); ax.set_xticklabels(cfg.LABELS, fontsize=10)
    ax.set_ylim(0, max(max(ece_i), max(ece_p)) * 1.52)
    ax.legend(fontsize=10, framealpha=1, edgecolor="#BBBBBB", loc="upper right")
    _style(ax, "Augmentation Strategy", "ECE (↓ better)  |  Δ = PH2 − ISIC")
    _save(fig, out_dir, "ece_bars.png")


# ── 3. Variance bar chart ─────────────────────────────────────────────────────
def plot_variance(isic, ph2, out_dir):
    ui  = [isic[s].avg_uncertainty.mean() for s in cfg.STRATEGIES]
    up  = [ph2[s].avg_uncertainty.mean()  for s in cfg.STRATEGIES]
    x, w = np.arange(5), 0.28
    fmt  = ".5f" if max(max(ui), max(up)) < 0.005 else ".4f"
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    ax.bar(x - w/2, ui, w, color=[c + "DD" for c in cfg.COLORS],
           edgecolor=DARK, lw=1.0, label="ISIC-2016", zorder=3)
    ax.bar(x + w/2, up, w, color=cfg.COLORS, edgecolor=DARK, lw=1.0,
           label="PH2 (OOD)", hatch=HATCH, zorder=3)
    _bar_labels(ax, x - w/2, ui, fmt=fmt, fs=8.5)
    _bar_labels(ax, x + w/2, up, fmt=fmt, fs=8.5)
    ax.set_xticks(x); ax.set_xticklabels(cfg.LABELS, fontsize=10)
    ax.set_ylim(0, max(max(ui), max(up)) * 1.28)
    ax.legend(fontsize=10, framealpha=1, edgecolor="#BBBBBB", loc="upper right")
    _style(ax, "Augmentation Strategy", "Mean Predictive Variance σ² (↓ better)")
    _save(fig, out_dir, "variance_bars.png")


# ── 4. Entropy bar chart ──────────────────────────────────────────────────────
def plot_entropy(isic, ph2, out_dir):
    ei  = [isic[s].pred_entropy.mean() for s in cfg.STRATEGIES]
    ep  = [ph2[s].pred_entropy.mean()  for s in cfg.STRATEGIES]
    x, w = np.arange(5), 0.28
    fmt  = ".5f" if max(max(ei), max(ep)) < 0.005 else ".4f"
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    ax.bar(x - w/2, ei, w, color=[c + "DD" for c in cfg.COLORS],
           edgecolor=DARK, lw=1.0, label="ISIC-2016", zorder=3)
    ax.bar(x + w/2, ep, w, color=cfg.COLORS, edgecolor=DARK, lw=1.0,
           label="PH2 (OOD)", hatch=HATCH, zorder=3)
    _bar_labels(ax, x - w/2, ei, fmt=fmt, fs=8.5)
    _bar_labels(ax, x + w/2, ep, fmt=fmt, fs=8.5)
    ax.set_xticks(x); ax.set_xticklabels(cfg.LABELS, fontsize=10)
    ax.set_ylim(0, max(max(ei), max(ep)) * 1.28)
    ax.legend(fontsize=10, framealpha=1, edgecolor="#BBBBBB", loc="upper right")
    _style(ax, "Augmentation Strategy", "Mean Predictive Entropy H (↓ better)")
    _save(fig, out_dir, "entropy_bars.png")


# ── 5. DSC boxplots ───────────────────────────────────────────────────────────
def _boxplot(data_dict, label, out_dir, fname):
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    data    = [data_dict[s].dice.values for s in cfg.STRATEGIES]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="white", lw=2.5),
                    whiskerprops=dict(color=DARK, lw=1.4),
                    capprops=dict(color=DARK, lw=1.4),
                    flierprops=dict(marker="o", markersize=4, alpha=0.45, markeredgewidth=0),
                    widths=0.55, zorder=3)
    for patch, color in zip(bp["boxes"], cfg.COLORS):
        patch.set_facecolor(color); patch.set_edgecolor(DARK)
        patch.set_linewidth(1.3); patch.set_alpha(0.90)
    kw = stats.kruskal(*data)
    ax.text(0.97, 0.03, f"Kruskal–Wallis  p = {kw.pvalue:.3f}",
            transform=ax.transAxes, ha="right", fontsize=9, color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(cfg.LABELS, fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    _style(ax, "Augmentation Strategy", "Dice Similarity Coefficient (DSC)")
    _save(fig, out_dir, fname)


# ── 6. Reliability diagrams (2×5 grid) ───────────────────────────────────────
def plot_reliability_grid(isic, ph2, out_dir):
    bins_e = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(2, 5, figsize=(W * 5, H * 2),
                             facecolor=BG, constrained_layout=True)
    row_labels = ["ISIC-2016", "PH2 (OOD)"]

    for col, s in enumerate(cfg.STRATEGIES):
        for row, dset in enumerate([isic, ph2]):
            ax  = axes[row][col]
            df  = dset[s]
            rng = df.pred_entropy.max() - df.pred_entropy.min() + 1e-8
            conf = 1.0 - (df.pred_entropy.values - df.pred_entropy.min()) / rng
            acc  = df.dice.values
            bc, ba = [], []
            for k in range(10):
                m = (conf >= bins_e[k]) & (conf < bins_e[k + 1])
                if m.sum() > 0:
                    bc.append(conf[m].mean()); ba.append(acc[m].mean())
            bc, ba = np.array(bc), np.array(ba)
            ax.plot([0, 1], [0, 1], color="#E05A5A", lw=1.5, ls="--", zorder=4)
            if len(bc) > 0:
                ax.bar(bc, ba, width=0.082, alpha=0.85,
                       color=cfg.COLORS[col], edgecolor=DARK, lw=0.6, zorder=3)
                ax.fill_between(bc, bc, ba, alpha=0.13, color="#B00000", zorder=2)
            ax.text(0.05, 0.88, f"ECE = {df.ece.mean():.3f}",
                    transform=ax.transAxes, fontsize=10, color=DARK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.9))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            _style(ax, fs=10)
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nAccuracy", fontsize=10, color=DARK)
            if row == 1:
                ax.set_xlabel("Confidence", fontsize=10)
            if row == 0:
                ax.set_title(cfg.LABELS[col], fontsize=11, fontweight="bold", color=DARK)
    _save(fig, out_dir, "reliability_diagrams.png")


# ── 7. Individual reliability diagrams (KDE-style) ───────────────────────────
def _reliability_one(df, color, fname, out_dir):
    bins_e = np.linspace(0, 1, 11)
    rng  = df.pred_entropy.max() - df.pred_entropy.min() + 1e-8
    conf = 1.0 - (df.pred_entropy.values - df.pred_entropy.min()) / rng
    acc  = df.dice.values
    bc, ba = [], []
    for k in range(10):
        m = (conf >= bins_e[k]) & (conf < bins_e[k + 1])
        if m.sum() > 0:
            bc.append(conf[m].mean()); ba.append(acc[m].mean())
    bc, ba = np.array(bc), np.array(ba)
    ba_smooth = gaussian_filter1d(ba, sigma=0.8) if len(bc) >= 3 else ba

    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.6, ls="--", zorder=3, label="Perfect calibration")
    if len(bc) > 1:
        ax.fill_between(bc, bc, ba_smooth, where=(ba_smooth >= bc),
                        alpha=0.18, color=color, zorder=2, label="Over-confident")
        ax.fill_between(bc, bc, ba_smooth, where=(ba_smooth < bc),
                        alpha=0.18, color="#CC3333", zorder=2, label="Under-confident")
        ax.plot(bc, ba_smooth, color=color, lw=2.5, zorder=4, label="Model calibration")
        ax.scatter(bc, ba, color=color, s=55, zorder=5, edgecolors=DARK, linewidths=0.8)
    ax.text(0.05, 0.91, f"ECE = {df.ece.mean():.3f}",
            transform=ax.transAxes, fontsize=11, color=DARK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9.5, framealpha=1, edgecolor="#CCCCCC", loc="lower right")
    _style(ax, "Confidence", "Accuracy")
    _save(fig, out_dir, fname)


# ── 8. Spearman scatter (one per strategy) ────────────────────────────────────
def plot_spearman_scatter(isic, out_dir):
    for i, s in enumerate(cfg.STRATEGIES):
        df  = isic[s]
        err = 1 - df.dice
        unc = df.avg_uncertainty
        rho, pval = stats.spearmanr(unc, err)
        fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
        ax.scatter(unc, err, alpha=0.38, s=16, color=cfg.COLORS[i], edgecolors="none", zorder=3)
        xs = np.linspace(unc.min(), unc.max(), 100)
        ax.plot(xs, np.poly1d(np.polyfit(unc, err, 1))(xs), color=DARK, lw=1.8, ls="--", zorder=4)
        pstr = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        ax.text(0.05, 0.93, f"ρ = {rho:.3f}\np {pstr}",
                transform=ax.transAxes, fontsize=10, color=DARK, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB", lw=0.9))
        _style(ax, "Predictive Variance σ²", "1 − DSC")
        _save(fig, out_dir, f"spearman_{s}.png")


# ── 9. Spearman ρ heatmap (4 metrics × 5 strategies) ─────────────────────────
def plot_spearman_heatmap(isic, ph2, out_dir):
    data = np.zeros((4, 5))
    for j, s in enumerate(cfg.STRATEGIES):
        err_i = 1 - isic[s].dice
        err_p = 1 - ph2[s].dice
        data[0, j] = stats.spearmanr(isic[s].avg_uncertainty, err_i).statistic
        data[1, j] = stats.spearmanr(isic[s].pred_entropy,    err_i).statistic
        data[2, j] = stats.spearmanr(ph2[s].avg_uncertainty,  err_p).statistic
        data[3, j] = stats.spearmanr(ph2[s].pred_entropy,     err_p).statistic

    row_labels = ["ISIC-2016 — σ²", "ISIC-2016 — H", "PH2 — σ²", "PH2 — H"]
    cmap  = LinearSegmentedColormap.from_list("spearman",
            ["#FFF8F0", "#F6C882", "#E07B3A", "#A63220", "#5C1010"], N=256)
    norm  = mcolors.Normalize(vmin=0.20, vmax=0.55)

    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=BG, constrained_layout=True)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for r in range(4):
        for c in range(5):
            v = data[r, c]
            ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if norm(v) > 0.55 else DARK)
    ax.set_xticks(range(5)); ax.set_xticklabels(cfg.LABELS,    fontsize=11, color=DARK)
    ax.set_yticks(range(4)); ax.set_yticklabels(row_labels, fontsize=11, color=DARK)
    ax.tick_params(length=0)
    ax.axhline(1.5, color="white", linewidth=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("Spearman ρ", fontsize=10, color=DARK)
    cbar.ax.tick_params(labelsize=9, colors=DARK)
    cbar.outline.set_visible(False)
    _save(fig, out_dir, "spearman_heatmap.png")


# ── 10. Bubble plots (DSC vs ECE, bubble size = variance) ────────────────────
def _bubble(data_dict, label, fname, out_dir):
    dv = [data_dict[s].dice.mean()           for s in cfg.STRATEGIES]
    ev = [data_dict[s].ece.mean()            for s in cfg.STRATEGIES]
    uv = [data_dict[s].avg_uncertainty.mean() * 1e6 for s in cfg.STRATEGIES]
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, constrained_layout=True)
    for i, s in enumerate(cfg.STRATEGIES):
        ax.scatter(dv[i], ev[i], s=max(uv[i] * 30, 150),
                   color=cfg.COLORS[i], alpha=0.90, edgecolors=DARK, lw=1.2, zorder=5)
        ax.annotate(cfg.LABELS[i], (dv[i], ev[i]),
                    xytext=(11, 6), textcoords="offset points",
                    fontsize=10, color=DARK, fontweight="bold")
    ax.invert_yaxis()
    ax.text(0.03, 0.04, "Bubble size ∝ σ²",
            transform=ax.transAxes, fontsize=9, color=ACCENT, style="italic")
    _style(ax, "Mean DSC (↑ better)", "ECE (↓ better)")
    _save(fig, out_dir, fname)


def main():
    args     = parse_args()
    out_dir  = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inference results...")
    isic, ph2 = load_data(args.results_root)
    print(f"ISIC: {len(isic[cfg.STRATEGIES[0]])} | PH2: {len(ph2[cfg.STRATEGIES[0]])} images\n")

    plot_dsc(isic, ph2, out_dir)
    plot_ece(isic, ph2, out_dir)
    plot_variance(isic, ph2, out_dir)
    plot_entropy(isic, ph2, out_dir)

    _boxplot(isic, "ISIC-2016", out_dir, "boxplot_isic.png")
    _boxplot(ph2,  "PH2 (OOD)", out_dir, "boxplot_ph2.png")

    plot_reliability_grid(isic, ph2, out_dir)
    for i, s in enumerate(cfg.STRATEGIES):
        _reliability_one(isic[s], cfg.COLORS[i], f"reliability_isic_{s}.png", out_dir)
        _reliability_one(ph2[s],  cfg.COLORS[i], f"reliability_ph2_{s}.png",  out_dir)

    plot_spearman_scatter(isic, out_dir)
    plot_spearman_heatmap(isic, ph2, out_dir)

    _bubble(isic, "ISIC-2016", "bubble_isic.png", out_dir)
    _bubble(ph2,  "PH2 (OOD)", "bubble_ph2.png",  out_dir)

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
