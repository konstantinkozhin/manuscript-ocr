import torch.nn as nn


class SEConv(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(self.pool(x))


class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SEBasicBlockLite(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = DWConvBlock(inplanes, planes, stride)
        self.conv2 = DWConvBlock(planes, planes, 1)
        self.se = SEConv(planes, reduction)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class SEResNet31Lite(nn.Module):
    def __init__(self, in_channels=3, out_channels=512, reduction=16):
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
            128, 256, blocks=1, stride=2, reduction=reduction
        )
        self.layer2 = self._make_layer(
            256, 256, blocks=2, stride=1, reduction=reduction
        )
        self.layer3 = self._make_layer(
            256, 512, blocks=5, stride=2, reduction=reduction
        )
        self.layer4 = self._make_layer(
            512, 512, blocks=3, stride=1, reduction=reduction
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=(2, 1), padding=1, groups=512, bias=False),
            nn.Conv2d(512, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.out_channels = out_channels

    # --------------------------------------------------------
    def _make_layer(self, inplanes, planes, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [SEBasicBlockLite(inplanes, planes, stride, downsample, reduction)]
        for _ in range(1, blocks):
            layers.append(SEBasicBlockLite(planes, planes, reduction=reduction))
        return nn.Sequential(*layers)

    # --------------------------------------------------------
    def forward(self, x):
        x = self.conv0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        return x
