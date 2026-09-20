"""SEResNet31Lite v2: signed residual branches before SE and residual addition."""

import torch.nn as nn

from .seresnetlite31 import SEResNet31Lite


class SEResNet31LiteV2(SEResNet31Lite):
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__(in_channels, out_channels)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                # Keep conv1 activation and the ReLU after residual addition.
                block.conv2.act = nn.Identity()
