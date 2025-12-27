"""Simple UNet stub for segmentation tasks."""
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(32, out_channels, 3, padding=1))

    def forward(self, x):
        e = self.encoder(x)
        out = self.decoder(e)
        return out
