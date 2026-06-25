from typing import Optional

import torch
import torch.nn as nn

from .._east.east import FeatureMergingBranchResNet, ResNetFeatureExtractor


class EASTV2OutputHead(nn.Module):
    """Three-head text instance output: text score, boundaries, centers."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 32):
        super().__init__()

        def make_head():
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )

        self.score_map = make_head()
        self.boundary_map = make_head()
        self.center_map = make_head()

    def forward(self, x):
        score = torch.sigmoid(self.score_map(x))
        boundary = torch.sigmoid(self.boundary_map(x))
        center = torch.sigmoid(self.center_map(x))
        return score, boundary, center


class EASTV2Model(nn.Module):
    """
    EAST-style detector for word-level instance segmentation.

    The model keeps EAST's ResNet feature pyramid and replaces the quad geometry
    output with three probability maps:
    score, boundary and center.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = False,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_first=freeze_first,
        )

        if backbone_name in {"resnet50", "resnet101"}:
            in_channels_list = (256, 512, 1024, 2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.decoder = FeatureMergingBranchResNet(in_channels_list=in_channels_list)
        self.output_head = EASTV2OutputHead()
        self.map_scale = 0.25

        if pretrained_model_path:
            state = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def forward(self, x):
        feats = self.backbone(x)
        merged = self.decoder(feats)
        score, boundary, center = self.output_head(merged)
        return {
            "score": score,
            "boundary": boundary,
            "center": center,
        }
