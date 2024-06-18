import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .blocks import SmallBasicBlock
from .stn import SpatialTransformerNetwork


class ICVLPR(nn.Module):
    """Indonesian Commercial Vehicle License Plate Recognition Model
    Inspired from LPRNet

    Args:
        num_classes (int): Number of classes the model take
        input_channels (int): Number of input channels for the image
    """

    def __init__(self,
                 num_classes: int = 37,
                 input_channels: int = 3,
                 use_stn: bool = True,
                 use_global_context: bool = True):
        super().__init__()
        assert input_channels in [1, 3], f'ICVLPR input_channels must be either 1 or 3, got {input_channels}'

        self.num_classes = num_classes
        self.global_context = use_global_context
        self.stn = use_stn

        self.stn_layer = SpatialTransformerNetwork()

        self.backbone = nn.ModuleDict({
            "conv_1": nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            "bn_1": nn.BatchNorm2d(64),
            "relu_1": nn.ReLU(),
            "max_pool_1": nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            "basic_block_1": SmallBasicBlock(in_channels=64, out_channels=128),
            "max_pool_2": nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            "basic_block_2": SmallBasicBlock(in_channels=128, out_channels=256),
            "basic_block_3": SmallBasicBlock(in_channels=256, out_channels=256),
            "max_pool_3": nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            "dropout_1": nn.Dropout2d(p=0.5),
            "conv_2": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1, padding=1),
            "bn_2": nn.BatchNorm2d(256),
            "relu_2": nn.ReLU(),
            "dropout_2": nn.Dropout2d(p=0.5),
        })

        self.pre_decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1, 13),
                      padding='same'),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

        self.gc_depth_adjust_layer = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stn:
            x = self.stn_layer(x)

        backbone_outputs = []
        for layer in self.backbone:
            x = self.backbone[layer](x)

            if self.global_context:
                backbone_outputs.append(x)

        if self.global_context:
            x = self.forward_global_context(backbone_outputs)

        x = self.pre_decoder(x)

        return x

    def forward_global_context(self, backbone_outputs: list) -> Tensor:
        inputs = [backbone_outputs[3], backbone_outputs[4], backbone_outputs[6], backbone_outputs[7]]
        outputs = []
        for i in inputs:
            outputs.append(F.adaptive_avg_pool2d(i, (3, 90)))

        x = backbone_outputs[-1]
        scale_5 = torch.div(x, x.square().mean())
        x = torch.cat([*outputs, scale_5], dim=1)

        x = self.gc_depth_adjust_layer(x)

        return x

    def use_global_context(self, active: bool = True) -> None:
        self.global_context = active
