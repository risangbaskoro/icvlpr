import torch

from torch import nn
from torch.nn import functional as F


class LocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d((16, 58))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=5, stride=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 32)
        self.fc2 = nn.Linear(32, 6)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv2(x))
        xs = torch.cat((x1, x2), dim=1).flatten(start_dim=1)
        xs = self.dropout(xs)
        xs = F.tanh(self.fc1(xs))
        theta = F.tanh(self.fc2(xs).view(-1, 2, 3))
        return theta


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = LocNet()

    def forward(self, x):
        theta = self.localization(x)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
