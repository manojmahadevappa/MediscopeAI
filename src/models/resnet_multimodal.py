"""Multimodal ResNet backbone and fusion module (minimal stub).
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNetMultimodal(nn.Module):
    def __init__(self, out_dim=2, pretrained=True):
        super().__init__()
        # two simple backbones (e.g., for CT and MRI)
        self.backbone_a = models.resnet18(pretrained=pretrained)
        self.backbone_a.fc = nn.Identity()
        self.backbone_b = models.resnet18(pretrained=pretrained)
        self.backbone_b.fc = nn.Identity()
        feat_dim = 512 + 512
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x_a, x_b):
        f_a = self.backbone_a(x_a)
        f_b = self.backbone_b(x_b)
        f = torch.cat([f_a, f_b], dim=1)
        return self.classifier(f)
