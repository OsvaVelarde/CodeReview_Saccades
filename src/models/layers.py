"""
References: 
"""
# ==============================================================

import torch.nn as nn
from torchvision.models.resnet import conv3x3, conv1x1

# ==============================================================

class BasicBlockV1(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(BasicBlockV1, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ==============================================================

class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = BN(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = BN(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

# ==============================================================

class Bottleneck(nn.Module):
    """
    Bottleneck version - Block
    """

    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BN(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BN(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BN(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ==============================================================
