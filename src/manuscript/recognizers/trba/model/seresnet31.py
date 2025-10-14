import torch.nn as nn
from torchvision.ops import DropBlock2d


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        reduction=16,
        dropblock_p=0.0,
        dropblock_block_size=5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

        self.dropblock = (
            DropBlock2d(p=dropblock_p, block_size=dropblock_block_size)
            if dropblock_p > 0
            else nn.Identity()
        )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out = self.dropblock(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class SEResNet31(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=512,
        reduction=16,
        dropblock_p=0.0,
        dropblock_block_size=5,
    ):
        super().__init__()
        # stem
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
        )

        # residual stages
        self.layer1 = self._make_layer(
            128,
            256,
            blocks=1,
            stride=2,
            reduction=reduction,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )
        self.layer2 = self._make_layer(
            256,
            256,
            blocks=2,
            stride=1,
            reduction=reduction,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )
        self.layer3 = self._make_layer(
            256,
            512,
            blocks=5,
            stride=2,
            reduction=reduction,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )
        self.layer4 = self._make_layer(
            512,
            512,
            blocks=3,
            stride=1,
            reduction=reduction,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(512, out_channels, 2, stride=(2, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.out_channels = out_channels

    def _make_layer(
        self,
        inplanes,
        planes,
        blocks,
        stride=1,
        reduction=16,
        dropblock_p=0.0,
        dropblock_block_size=5,
    ):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [
            SEBasicBlock(
                inplanes,
                planes,
                stride,
                downsample,
                reduction,
                dropblock_p=dropblock_p,
                dropblock_block_size=dropblock_block_size,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                SEBasicBlock(
                    planes,
                    planes,
                    reduction=reduction,
                    dropblock_p=dropblock_p,
                    dropblock_block_size=dropblock_block_size,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        return x  # [B, 512, H≈1, W/4]
