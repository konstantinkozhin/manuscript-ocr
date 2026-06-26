from typing import Optional

import torch
import torch.nn as nn

from .._east.east import FeatureMergingBranchResNet, ResNetFeatureExtractor


class PSEOutputHead(nn.Module):
    """PSENet-style head: text map + progressively shrunk kernel maps."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, kernel_num: int = 7):
        super().__init__()
        self.kernel_num = int(kernel_num)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.kernel_num, kernel_size=1),
        )

    def forward(self, x):
        logits = self.conv(x)
        maps = torch.sigmoid(logits)
        return logits, maps


class PSEModel(nn.Module):
    """EAST feature pyramid with a PSENet/PSE detection head."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = False,
        pretrained_model_path: Optional[str] = None,
        kernel_num: int = 7,
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
        self.output_head = PSEOutputHead(kernel_num=kernel_num)
        self.kernel_num = int(kernel_num)
        self.map_scale = 0.25

        if pretrained_model_path:
            state = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def forward(self, x):
        feats = self.backbone(x)
        merged = self.decoder(feats)
        logits, maps = self.output_head(merged)
        return {
            "logits": logits,
            "maps": maps,
            "text": maps[:, :1],
            "kernels": maps[:, 1:],
        }
