import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        return x


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_first: bool = False,
    ):
        super().__init__()
        # select model and weights
        if backbone_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        elif backbone_name == "resnet101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # optional freezing of initial layers
        if freeze_first:
            for name, param in model.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1")):
                    param.requires_grad = False

        # extract feature maps at 4 stages
        self.extractor = create_feature_extractor(
            model,
            return_nodes={
                "layer1": "res1",  # stride 4
                "layer2": "res2",  # stride 8
                "layer3": "res3",  # stride 16
                "layer4": "res4",  # stride 32
            },
        )

    def forward(self, x):
        return self.extractor(x)


class FeatureMergingBranchResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DecoderBlock(in_channels=2048, mid_channels=512, out_channels=512)
        self.block2 = DecoderBlock(
            in_channels=512 + 1024, mid_channels=256, out_channels=256
        )
        self.block3 = DecoderBlock(
            in_channels=256 + 512, mid_channels=128, out_channels=128
        )
        self.block4 = DecoderBlock(
            in_channels=128 + 256, mid_channels=64, out_channels=32
        )

    def forward(self, feats):
        f1, f2, f3, f4 = feats["res1"], feats["res2"], feats["res3"], feats["res4"]
        h4 = self.block1(f4)
        h4_up = F.interpolate(h4, scale_factor=2, mode="bilinear", align_corners=False)
        h3 = self.block2(torch.cat([h4_up, f3], dim=1))
        h3_up = F.interpolate(h3, scale_factor=2, mode="bilinear", align_corners=False)
        h2 = self.block3(torch.cat([h3_up, f2], dim=1))
        h2_up = F.interpolate(h2, scale_factor=2, mode="bilinear", align_corners=False)
        h1 = self.block4(torch.cat([h2_up, f1], dim=1))
        return h1


class OutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_map = nn.Conv2d(32, 1, kernel_size=1)
        self.geo_map = nn.Conv2d(32, 8, kernel_size=1)

    def forward(self, x):
        score = torch.sigmoid(self.score_map(x))
        geometry = self.geo_map(x)
        return score, geometry


class EAST(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = False,
        pretrained_model_path: str = None,
    ):
        super().__init__()
        # backbone with choice
        self.backbone = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_first=freeze_first,
        )
        self.decoder = FeatureMergingBranchResNet()
        self.output_head = OutputHead()
        # scales for maps (legacy behavior)
        self.score_scale = 0.25
        self.geo_scale = 0.25

        # optional load weights
        if pretrained_model_path:
            state = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"Loaded pretrained model from {pretrained_model_path}")

    def forward(self, x):
        feats = self.backbone(x)
        merged = self.decoder(feats)
        score, geometry = self.output_head(merged)
        return {"score": score, "geometry": geometry}
