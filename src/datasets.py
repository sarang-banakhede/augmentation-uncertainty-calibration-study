import numpy as np
from pathlib import Path
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class SkinDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.ids       = sorted([p.stem for p in self.img_dir.glob("*.png")])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        img  = np.array(Image.open(self.img_dir / f"{name}.png").convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{name}.png").convert("L"))
        mask = mask.astype(np.float32) if mask.max() <= 1 else (mask > 127).astype(np.float32)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"].float()
            mask = aug["mask"].unsqueeze(0).float()

        return img, mask, name


class PH2Dataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, image_size: int):
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.ids       = sorted([p.stem for p in self.img_dir.glob("*.bmp")])
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        img  = np.array(Image.open(self.img_dir  / f"{name}.bmp").convert("RGB"))
        mask = np.array(Image.open(self.label_dir / f"{name}_lesion.bmp").convert("L"))
        mask = (mask > 127).astype(np.float32)

        aug  = self.transform(image=img, mask=mask)
        return aug["image"].float(), aug["mask"].unsqueeze(0).float(), name
