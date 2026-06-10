import numpy as np
import torch
import torch.nn as nn

from src.augmentations import mixup_batch, cutmix_batch
from src.metrics import batch_metrics


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_inference(
    model: nn.Module,
    imgs: torch.Tensor,
    mc_passes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            preds.append(model(imgs.to(device)).cpu().numpy())
    preds = np.stack(preds, axis=0)   # [T, B, 1, H, W]
    return preds.mean(axis=0), preds.var(axis=0)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    strategy: str,
    device: torch.device,
) -> dict:
    model.train()
    accum = {k: 0.0 for k in ("loss", "dice", "precision", "recall", "f1", "accuracy")}
    n = 0

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        if strategy == "mixing":
            if np.random.rand() < 0.5:
                imgs, masks = mixup_batch(imgs, masks)
            else:
                imgs, masks = cutmix_batch(imgs, masks)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        m = batch_metrics(preds.detach().cpu(), masks.detach().cpu())
        accum["loss"] += loss.item()
        for k in m:
            accum[k] += m[k]
        n += 1

    return {k: v / n for k, v in accum.items()}


def eval_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    accum = {k: 0.0 for k in ("loss", "dice", "precision", "recall", "f1", "accuracy")}
    n = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            m = batch_metrics(preds.cpu(), masks.cpu())
            accum["loss"] += loss.item()
            for k in m:
                accum[k] += m[k]
            n += 1

    return {k: v / n for k, v in accum.items()}
