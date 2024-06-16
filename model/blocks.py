import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class SmallBasicBlock(nn.Module):
    """Small Basic Block for Residual
    Inspired from Squeeze Fire Blocks and Inception Blocks

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_div4 = out_channels // 4

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_div4, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            # TODO: Check the stride h and pad h for Conv2d below.
            nn.Conv2d(in_channels=out_div4, out_channels=out_div4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            # TODO: Check the stride w and pad w for Conv2d below.
            nn.Conv2d(in_channels=out_div4, out_channels=out_div4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=out_div4, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x