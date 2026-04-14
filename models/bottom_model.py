"""
Bottom models for passive parties.

- MLP for tabular data
- CNN (real convolutional) for image data
- ResNet-style for image data (higher capacity)

Note: In VFL, each party receives a PARTITION of the image (e.g., left/right half).
The input to CNN/ResNet models is (N, C, H, W_i) where W_i < W.
We keep the spatial structure instead of flattening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# MLP for tabular data
# ======================================================================

class BottomModelMLP(nn.Module):
    """MLP bottom model for tabular / flattened features."""

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# True CNN for image data
# ======================================================================

class BottomModelCNN(nn.Module):
    """
    Real CNN bottom model for image partitions.

    Input: flattened image partition (N, D).
    We infer the spatial shape and apply convolutions.
    """

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim

        # We'll use an adaptive architecture:
        # For small inputs (D < 512): use MLP (not enough spatial structure)
        # For larger inputs: use real conv layers
        if input_dim >= 512:
            # Assume input is roughly C*H*W, use conv layers
            self.use_conv = True
            self.conv = nn.Sequential(
                nn.LazyConv2d(32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.LazyConv2d(64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.fc = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.use_conv = False
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
            )

        # Store shape info (will be set on first forward)
        self._img_shape = None

    def _infer_shape(self, D):
        """Infer (C, H, W) from flattened dim D."""
        # Common VFL image partition shapes:
        # CIFAR-10 half: 3*32*16 = 1536
        # CIFAR-10 full channel: 1*32*32 = 1024 or 2*32*32 = 2048
        # MNIST half: 1*28*14 = 392
        for C in [3, 1, 2]:
            for H in [32, 28, 64]:
                if D % (C * H) == 0:
                    W = D // (C * H)
                    if 4 <= W <= 64:
                        return (C, H, W)
        # Fallback: treat as 1-channel square-ish
        import math
        side = int(math.sqrt(D))
        if side * side == D:
            return (1, side, side)
        # Last resort: 1 x D x 1 (will be basically 1D conv)
        return (1, 1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return self.net(x)

        # Reshape flat input to image
        if self._img_shape is None:
            self._img_shape = self._infer_shape(x.shape[1])

        C, H, W = self._img_shape
        x_img = x.view(x.size(0), C, H, W)
        feat = self.conv(x_img)
        return self.fc(feat)


# ======================================================================
# ResNet-style for image data (higher capacity)
# ======================================================================

class ResidualBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class BottomModelResNet(nn.Module):
    """
    Small ResNet bottom model for image partitions.
    Architecture: Conv -> 2 ResBlocks -> Pool -> FC
    """

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self._img_shape = None

        self.stem = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def _infer_shape(self, D):
        for C in [3, 1, 2]:
            for H in [32, 28, 64]:
                if D % (C * H) == 0:
                    W = D // (C * H)
                    if 4 <= W <= 64:
                        return (C, H, W)
        import math
        side = int(math.sqrt(D))
        if side * side == D:
            return (1, side, side)
        return (1, 1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._img_shape is None:
            self._img_shape = self._infer_shape(x.shape[1])

        C, H, W = self._img_shape
        x_img = x.view(x.size(0), C, H, W)

        out = self.stem(x_img)
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out)
        return self.fc(out)


# ======================================================================
# Factory
# ======================================================================

def build_bottom_model(
    model_type: str, input_dim: int, output_dim: int = 128
) -> nn.Module:
    """Factory for bottom models."""
    if model_type == "mlp":
        return BottomModelMLP(input_dim, output_dim)
    elif model_type == "cnn":
        return BottomModelCNN(input_dim, output_dim)
    elif model_type == "resnet":
        return BottomModelResNet(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown bottom model type: {model_type}. "
                         f"Choose from: mlp, cnn, resnet")
