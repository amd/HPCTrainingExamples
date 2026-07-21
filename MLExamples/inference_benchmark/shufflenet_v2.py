import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

def conv3x3(in_channels, out_channels, stride, padding=1, groups=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=padding,
                    groups=groups,
                    bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                    kernel_size=1, stride=stride,padding=0,
                    bias=False)

class ShufflenetUnit(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShufflenetUnit, self).__init__()
        self.downsample = downsample

        if not self.downsample: #---if not downsample, then channel split, so the channel become half
            inplanes = inplanes // 2
            planes = planes // 2
 
        self.conv1x1_1 = conv1x1(in_channels=inplanes, out_channels=planes)
        self.conv1x1_1_bn = nn.BatchNorm2d(planes)

        self.dwconv3x3 = conv3x3(in_channels=planes, out_channels=planes, stride=stride, groups=planes)
        self.dwconv3x3_bn= nn.BatchNorm2d(planes)

        self.conv1x1_2 = conv1x1(in_channels=planes, out_channels=planes)
        self.conv1x1_2_bn = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def _channel_split(self, features, ratio=0.5):
        """
        ratio: c'/c, default value is 0.5
        """ 
        size = features.size()[1]
        split_idx = int(size * ratio)
        return features[:,:split_idx], features[:,split_idx:]

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        if  self.downsample:
            #x1 = x.clone() #----deep copy x, so where x2 is modified, x1 not be affected
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)

        #----right branch----- 
        x2 = self.conv1x1_1(x2)
        x2 = self.conv1x1_1_bn(x2)
        x2 = self.relu(x2)
         
        x2 = self.dwconv3x3(x2)
        x2 = self.dwconv3x3_bn(x2)
    
        x2 = self.conv1x1_2(x2)
        x2 = self.conv1x1_2_bn(x2)
        x2 = self.relu(x2)

        #---left branch-------
        if self.downsample:
            x1 = self.downsample(x1)

        x = torch.cat([x1, x2], 1)
        x = self._channel_shuffle(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, feature_dim, layers_num, num_classes=1000):
        super(ShuffleNet, self).__init__()
        dim1, dim2, dim3, dim4, dim5 = feature_dim
        self.conv1 = conv3x3(in_channels=3, out_channels=dim1, 
                            stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_layer(dim1, dim2, layers_num[0]) 
        self.stage3 = self._make_layer(dim2, dim3, layers_num[1])
        self.stage4 = self._make_layer(dim3, dim4, layers_num[2])

        self.conv5 = conv1x1(in_channels=dim4, out_channels=dim5)
        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(dim5, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """

    def _make_layer(self, dim1, dim2, blocks_num):
        half_channel = dim2 // 2
        downsample = nn.Sequential(
            conv3x3(in_channels=dim1, out_channels=dim1, stride=2, padding=1, groups=dim1),
            nn.BatchNorm2d(dim1),
            conv1x1(in_channels=dim1, out_channels=half_channel),
            nn.BatchNorm2d(half_channel),
            nn.ReLU(inplace=True)
        )

        layers = []
        layers.append(ShufflenetUnit(dim1, half_channel, stride=2, downsample=downsample))
        for i in range(blocks_num):
            layers.append(ShufflenetUnit(dim2, dim2, stride=1))

        return nn.Sequential(*layers) 

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("x0.size:\t", x.size())
        x = self.maxpool(x)         
        #print("x1.size:\t", x.size())
        x = self.stage2(x)
        #print("x2.size:\t", x.size())
        x = self.stage3(x)
        #print("x3.size:\t", x.size())
        x = self.stage4(x)
        #print("x4.size:\t", x.size())
        
        x = self.conv5(x)
        #print("x5.size:\t", x.size())
        x = self.globalpool(x)
        #print("x6.size:\t", x.size())

        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

features = {
    "0.5x":[24, 48, 96, 192, 1024],
    "1x":[24, 116, 232, 464, 1024],
    "1.5x":[24, 176, 352, 704, 1024],
    "2x":[24, 244, 488, 976, 2048]
}

def shufflenet():
    model = ShuffleNet(features["1x"], [3, 7, 3])
    return model 

if __name__=="__main__":
    model = shufflenet().cuda()
    print(model)
    x = torch.rand((1,3,224,224))
    x = Variable(x).cuda()
    x = model(x)
