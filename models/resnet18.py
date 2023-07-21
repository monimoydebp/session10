import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def perform_norm(type, num_channels, channel_size_w, channel_size_h, num_groups=2):
        if type == 'BN':
            return nn.BatchNorm2d(num_channels)
        elif type == 'LN':
            return nn.LayerNorm((num_channels, channel_size_w, channel_size_h))
        elif type == 'GN':
            return nn.GroupNorm(num_groups, num_channels)
        

    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(BasicBlock, self).__init__()
        self.dropout_prob=p
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop_out1 = nn.Dropout2d(p=self.dropout_prob)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop_out2 = nn.Dropout2d(p=self.dropout_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(             
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout2d(p=self.dropout_prob)
            )

    def forward(self, x):
        out = F.relu(self.drop_out1(self.bn1(self.conv1(x))))
        out = self.drop_out2(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, p=0.0, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.p = p

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2 )
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)


def ResNet18(p=0.0):
    return ResNet(BasicBlock, [2,2,2,2], p)
