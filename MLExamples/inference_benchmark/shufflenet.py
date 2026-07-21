import os
import sys
import torch
import torch.nn as nn
import math
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ShufflenetUnit(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, flag=False):
        super(ShufflenetUnit, self).__init__()
        self.downsample = downsample
        group_num = 3
        self.flag = flag
        if self.flag:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=group_num, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, groups=group_num, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def _shuffle(self, features, g):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.flag:
            out = self._shuffle(out, 3)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            out = torch.cat((out, residual), 1) 
        else:
            out += residual
        out = self.relu(out)

        return out

class ShuffleNet(nn.Module):
    inplanes = 24
    def __init__(self, block, layers, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3,
                               padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.stage2 = self._make_layer(block, 240, layers[0], True) 
        self.stage3 = self._make_layer(block, 480, layers[1], False)
        self.stage4 = self._make_layer(block, 960, layers[2], False)

        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(960, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, flag):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2,padding=1)
        )

        inner_plane = (planes - self.inplanes) / 4
        layers = []
        layers.append(block(self.inplanes, inner_plane, 2, downsample, flag=flag))
        self.inplanes = planes
        for i in range(blocks):
            layers.append(block(planes, planes/4))

        return nn.Sequential(*layers) 

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def shufflenet():
    model = ShuffleNet(ShufflenetUnit, [3, 7, 3])
    return model 

if __name__=="__main__":
    model = shufflenet()
    print(model)
