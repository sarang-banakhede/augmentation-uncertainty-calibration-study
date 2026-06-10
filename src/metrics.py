import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def batch_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    eps = 1e-6
    p   = (preds > 0.5).float()
    t   = targets.float()
    tp  = (p * t).sum()
    fp  = (p * (1 - t)).sum()
    fn  = ((1 - p) * t).sum()
    tn  = ((1 - p) * (1 - t)).sum()
    pr  = (tp + eps) / (tp + fp + eps)
    rc  = (tp + eps) / (tp + fn + eps)
    return {
        "dice":      ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),
        "precision": pr.item(),
        "recall":    rc.item(),
        "f1":        (2 * pr * rc / (pr + rc + eps)).item(),
        "accuracy":  ((tp + tn) / (tp + fp + fn + tn + eps)).item(),
    }


def per_image_metrics(pred_np: np.ndarray, gt_np: np.ndarray) -> dict:
    eps      = 1e-6
    pred_bin = (pred_np > 0.5).astype(np.float32)
    tp = (pred_bin * gt_np).sum()
    fp = (pred_bin * (1 - gt_np)).sum()
    fn = ((1 - pred_bin) * gt_np).sum()
    tn = ((1 - pred_bin) * (1 - gt_np)).sum()
    return {
        "dice":      float((2 * tp + eps) / (pred_bin.sum() + gt_np.sum() + eps)),
        "iou":       float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall":    float((tp + eps) / (tp + fn + eps)),
        "accuracy":  float((tp + tn) / (tp + fp + fn + tn + eps)),
    }


def compute_hd95(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    pred_b = pred_np > 0.5
    gt_b   = gt_np   > 0.5
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("nan")
    pd = distance_transform_edt(~pred_b)
    gd = distance_transform_edt(~gt_b)
    return float(np.percentile(np.concatenate([pd[gt_b], gd[pred_b]]), 95))


def compute_ece(probs_list: list, labels_list: list, n_bins: int = 10) -> float:
    probs  = np.concatenate(probs_list).flatten()
    labels = np.concatenate(labels_list).flatten()
    bins   = np.linspace(0, 1, n_bins + 1)
    ece, N = 0.0, len(probs)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / N) * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


def compute_brier(probs_list: list, labels_list: list) -> float:
    probs  = np.concatenate(probs_list).flatten()
    labels = np.concatenate(labels_list).flatten()
    return float(np.mean((probs - labels) ** 2))


def compute_predictive_entropy(mean_pred: np.ndarray) -> float:
    p = np.clip(mean_pred, 1e-8, 1 - 1e-8)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())
