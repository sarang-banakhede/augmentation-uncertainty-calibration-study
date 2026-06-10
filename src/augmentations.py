import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

_NORM = [
    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
    ToTensorV2(),
]


def get_transforms(strategy: str, image_size: int):
    resize = A.Resize(image_size, image_size)

    if strategy == "baseline":
        train_tf = A.Compose([resize, *_NORM])

    elif strategy == "geometric":
        train_tf = A.Compose([
            resize,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            *_NORM,
        ])

    elif strategy == "intensity":
        train_tf = A.Compose([
            resize,
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            *_NORM,
        ])

    elif strategy == "mixing":
        train_tf = A.Compose([resize, A.HorizontalFlip(p=0.5), *_NORM])

    elif strategy == "occlusion":
        train_tf = A.Compose([
            resize,
            A.HorizontalFlip(p=0.5),
            A.GridDropout(ratio=0.3, p=0.4),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.4),
            *_NORM,
        ])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    test_tf = A.Compose([resize, *_NORM])
    return train_tf, test_tf


def mixup_batch(imgs: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0))
    return lam * imgs + (1 - lam) * imgs[idx], lam * masks + (1 - lam) * masks[idx]


def cutmix_batch(imgs: torch.Tensor, masks: torch.Tensor, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0))
    _, _, H, W = imgs.shape
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    imgs[:, :, y1:y2, x1:x2]  = imgs[idx, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = masks[idx, :, y1:y2, x1:x2]
    return imgs, masks
