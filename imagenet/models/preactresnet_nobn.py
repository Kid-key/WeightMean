
"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(
            #nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=True),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1,bias=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)
        
    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual = nn.Sequential(
            #nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride,bias=True),

            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=True),

            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1,bias=True)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)
    
    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=1,bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layers(block, num_block[0], 64,  1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(self.input_channels, num_classes)
        self.lastbn = nn.BatchNorm1d(num_classes,affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                if m.in_channels==3:
                    m.weight.data.normal_(0, math.sqrt(2. / n))    
                else:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.lastbn(x)


        return x

def preactresnet18_cbn():
    return PreActResNet(PreActBasic, [2, 2, 2, 2])
    
def preactresnet34():
    return PreActResNet(PreActBasic, [3, 4, 6, 3])

def preactresnet50_cbn():
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])

def preactresnet101_cbn():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])

def preactresnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])

