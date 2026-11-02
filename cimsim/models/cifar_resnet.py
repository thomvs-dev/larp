"""CIFAR ResNet-20/32/44/56 with the same module names as chenyaofo/pytorch-cifar-models,
so its pretrained state dicts load directly (no torch.hub code execution needed).

Pretrained CIFAR-10 ResNet-20 (top-1 92.60% reported by the source repo):
https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt
"""

from __future__ import annotations

import torch
import torch.nn as nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
PRETRAINED = {
    "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt",
}


def conv3x3(i: int, o: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False)


def conv1x1(i: int, o: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(i, o, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CifarResNet(nn.Module):
    def __init__(self, layers: list[int], num_classes: int = 10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet([3, 3, 3], num_classes)


def load_pretrained_resnet20() -> CifarResNet:
    model = resnet20()
    sd = torch.hub.load_state_dict_from_url(PRETRAINED["resnet20"], map_location="cpu", check_hash=True,
                                            weights_only=True)
    model.load_state_dict(sd)
    return model.eval()
