from torch import nn
import torch


class BaseConv(nn.Module):

    def __init__(self, inc, ouc, k, s):
        super().__init__()
        self.conv = nn.Conv2d(inc, ouc, k, s, (k-1)//2, 1, 1, False)
        self.norm = nn.BatchNorm2d(ouc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    
    def __init__(self, inc, ouc,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False):
        super().__init__()
        hidc = int(inc * expansion)
        # Conv = DWConv if depthwise else BaseConv
        Conv = BaseConv
        self.conv1 = BaseConv(inc, ouc, 1, s=1)
        self.conv2 = Conv(hidc, ouc, 3, s=1)
        self.use_add = shortcut and inc == ouc

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):

    def __init__(self, inc, ouc, depth=1, shortcut=True,
                 expand=0.5, depthwise=False):
        super().__init__()
        hidc = int(ouc * expand)
        self.conv1 = BaseConv(inc, hidc, 1, 1)
        self.conv2 = BaseConv(inc, hidc, 1, 1)
        self.conv3 = BaseConv(2 * hidc, ouc, 1, 1)
        module_list = [
            Bottleneck(hidc, hidc, shortcut, 1.0,
                       depthwise) for _ in range(depth)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

