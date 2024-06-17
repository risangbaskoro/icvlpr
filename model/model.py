import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from blocks import (
    GlobalContextBlock,
    SmallBasicBlock
)


class ICVLPR(nn.Module):
    """Indonesian Commercial Vehicle License Plate Recognition Model
    Inspired from LPRNet

    Args:
        num_classes (int): Number of classes the model take
        input_channels (int): Number of input channels for the image
    """

    def __init__(self,
                 num_classes: int = 37,
                 input_channels: int = 3):
        super().__init__()
        assert input_channels in [1, 3]
        self.num_classes = num_classes

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

        self.gc_depth_adjust = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=256, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pre_decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1, 13),
                      padding='same'),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gc_outputs = []

        for layer in self.backbone:
            x = self.backbone[layer](x)
            if layer in ["relu_1", "basic_block_1", "basic_block_2", "basic_block_3"]:
                gc_outputs.append(F.adaptive_avg_pool2d(x, (3, 90)))

        scale_5 = torch.div(x, x.square().mean())

        x = torch.concat([*gc_outputs, scale_5], dim=1)
        x = self.gc_depth_adjust(x)

        x = self.pre_decoder(x)

        return x


if __name__ == '__main__':
    sample_img = torch.rand(1, 3, 24, 94)
    model = ICVLPR()

    result = model(sample_img)
    print(result.shape)
