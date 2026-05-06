from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR images."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

