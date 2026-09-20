"""Optional thin-plate-spline rectification using only PyTorch.

Adapted from clovaai/deep-text-recognition-benchmark, modules/transformation.py
https://github.com/clovaai/deep-text-recognition-benchmark
Upstream license: Apache-2.0;
"""

import torch
from torch import nn
from torch.nn import functional as F


class TPSGridGenerator(nn.Module):
    """Map fiducial displacements [B, K, 2] to sampling grids [B, H, W, 2]."""

    def __init__(self, image_size, num_fiducial=20):
        super().__init__()
        if num_fiducial < 4 or num_fiducial % 2:
            raise ValueError("TPS num_fiducial must be even and at least 4")
        self.height, self.width = image_size
        if min(image_size) < 8:
            raise ValueError("TPS image height and width must be at least 8")
        x = torch.linspace(-1, 1, num_fiducial // 2, dtype=torch.float64)
        control = torch.cat(
            [
                torch.stack((x, -torch.ones_like(x)), dim=1),
                torch.stack((x, torch.ones_like(x)), dim=1),
            ]
        )
        gy = (2 * torch.arange(self.height, dtype=torch.float64) + 1) / self.height - 1
        gx = (2 * torch.arange(self.width, dtype=torch.float64) + 1) / self.width - 1
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

        def radial(a, b):
            r2 = (a[:, None] - b[None]).square().sum(dim=-1)
            # U(r) = r^2 log(r), with the continuous value U(0) = 0.
            return 0.5 * r2 * r2.clamp_min(torch.finfo(r2.dtype).tiny).log()

        affine = torch.cat(
            (torch.ones(num_fiducial, 1, dtype=control.dtype), control), dim=1
        )
        system = torch.zeros(num_fiducial + 3, num_fiducial + 3, dtype=control.dtype)
        system[:num_fiducial, :num_fiducial] = radial(control, control)
        system[:num_fiducial, num_fiducial:] = affine
        system[num_fiducial:, :num_fiducial] = affine.T
        rhs = torch.zeros(num_fiducial + 3, num_fiducial, dtype=control.dtype)
        rhs[:num_fiducial] = torch.eye(num_fiducial, dtype=control.dtype)
        basis = torch.cat(
            (radial(points, control), torch.ones_like(points[:, :1]), points), dim=1
        )
        interpolation = basis @ torch.linalg.solve(system, rhs)
        self.register_buffer("control_points", control.float())
        self.register_buffer("base_grid", points.float())
        self.register_buffer("interpolation", interpolation.float())

    def forward(self, offsets):
        # Autocast would otherwise lower matmul precision even with FP32 inputs.
        with torch.autocast(device_type=offsets.device.type, enabled=False):
            points = self.base_grid.float() + torch.matmul(
                self.interpolation.float(), offsets.float()
            )
            return points.reshape(offsets.shape[0], self.height, self.width, 2)


class TPSLocalizationNetwork(nn.Module):
    """Predict offsets from two rows of canonical fiducials; initially all zero."""

    def __init__(self, in_channels=3, num_fiducial=20):
        super().__init__()
        self.num_fiducial = num_fiducial
        layers = []
        for index, channels in enumerate((64, 128, 256, 512)):
            layers.extend(
                [
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 3:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(256, num_fiducial * 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, images):
        features = self.fc1(self.conv(images).flatten(1))
        with torch.autocast(device_type=images.device.type, enabled=False):
            return self.fc2(features.float()).reshape(
                images.shape[0], self.num_fiducial, 2
            )


class TPSRectifier(nn.Module):
    # Source: https://github.com/clovaai/deep-text-recognition-benchmark (Apache-2.0).
    """Learn image rectification jointly with the recognizer; no point labels."""

    def __init__(self, image_size, in_channels=3, num_fiducial=20):
        super().__init__()
        self.grid_generator = TPSGridGenerator(image_size, num_fiducial)
        self.localization = TPSLocalizationNetwork(in_channels, num_fiducial)

    def forward(self, images):
        offsets = self.localization(images)
        grid = self.grid_generator(offsets)
        with torch.autocast(device_type=images.device.type, enabled=False):
            rectified = F.grid_sample(
                images.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        return rectified.to(images.dtype)
