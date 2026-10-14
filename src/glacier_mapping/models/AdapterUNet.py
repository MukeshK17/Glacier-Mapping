import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights, ResNet34_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class InputAdapterStem(nn.Module):
    """
    Learned 1x1 projection adapter from 18 multi-spectral/geophysical channels
    to 3 channels for ImageNet-pretrained encoders.
    Initializes optical RGB channels with an identity mapping and
    geophysical/SAR channels with small variance.
    """

    def __init__(self, in_channels: int = 18, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            # Initialize first 3 channels (Optical RGB/NIR) to preserve natural contrast
            for i in range(min(out_channels, in_channels, 3)):
                self.proj.weight[i, i, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.proj(x)))

    def get_band_importances(self) -> torch.Tensor:
        """
        Computes normalized L2-norm of the projection weights across input bands.
        Provides a direct, mathematically principled metric of band contribution.
        """
        weights = self.proj.weight.squeeze(-1).squeeze(-1)  # [out_channels, in_channels]
        l2_norms = torch.norm(weights, p=2, dim=0)  # [in_channels]
        return l2_norms / (l2_norms.sum() + 1e-8)


class ConvBlock(nn.Module):
    """Standard double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsampling decoder block with skip connection concatenation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StandaloneResidualEncoder(nn.Module):
    """Pure PyTorch fallback residual encoder when torchvision is not loaded."""

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def make_layer(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.layer1 = make_layer(64, 64)
        self.layer2 = make_layer(64, 128, stride=2)
        self.layer3 = make_layer(128, 256, stride=2)
        self.layer4 = make_layer(256, 512, stride=2)

    def forward(self, x: torch.Tensor):
        x0 = self.init_conv(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class AdapterUNet(nn.Module):
    """
    Multi-Band Glacier Segmentation Network:
    - InputAdapterStem: maps 18 channels -> 3 channels with learned projection
    - Pretrained Backbone Encoder (ResNet18 / ResNet34)
    - Feature Pyramid U-Net Decoder with Skip Connections
    """

    def __init__(
        self,
        in_channels: int = 18,
        num_classes: int = 1,
        backbone: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.adapter = InputAdapterStem(in_channels=in_channels, out_channels=3)

        # Build Encoder
        if HAS_TORCHVISION:
            if backbone == "resnet18":
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                base = models.resnet18(weights=weights)
            else:
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                base = models.resnet34(weights=weights)

            self.init_conv = nn.Sequential(base.conv1, base.bn1, base.relu)
            self.maxpool = base.maxpool
            self.layer1 = base.layer1  # 64 channels
            self.layer2 = base.layer2  # 128 channels
            self.layer3 = base.layer3  # 256 channels
            self.layer4 = base.layer4  # 512 channels
            self.is_custom = False
        else:
            self.custom_encoder = StandaloneResidualEncoder()
            self.is_custom = True

        # U-Net Decoder
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Project 18 channels to 3
        x_3ch = self.adapter(x)

        # Step 2: Encoder pass
        if self.is_custom:
            x0, x1, x2, x3, x4 = self.custom_encoder(x_3ch)
        else:
            x0 = self.init_conv(x_3ch)       # [B, 64, H/2, W/2]
            x_pool = self.maxpool(x0)        # [B, 64, H/4, W/4]
            x1 = self.layer1(x_pool)         # [B, 64, H/4, W/4]
            x2 = self.layer2(x1)             # [B, 128, H/8, W/8]
            x3 = self.layer3(x2)             # [B, 256, H/16, W/16]
            x4 = self.layer4(x3)             # [B, 512, H/32, W/32]

        # Step 3: Decoder with skip connections
        d4 = self.dec4(x4, x3)               # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, x2)               # [B, 128, H/8, W/8]
        d2 = self.dec2(d3, x1)               # [B, 64, H/4, W/4]
        d1 = self.dec1(d2, x0)               # [B, 64, H/2, W/2]

        # Step 4: Final output
        logits = self.final_upsample(d1)     # [B, 1, H, W]

        # Ensure exact spatial match to input
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        return logits

    def get_band_importances(self) -> torch.Tensor:
        return self.adapter.get_band_importances()
