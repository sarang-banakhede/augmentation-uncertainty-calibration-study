import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        features: list = [64, 128, 256, 512],
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()

        for f in features:
            self.downs.append(DoubleConv(in_ch, f, dropout_p))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f, dropout_p))

        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x     = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))
