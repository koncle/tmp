import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """ Pre-activation version of the BasicBlock. """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64):
        super(PreActResNet, self).__init__()
        self.in_planes = model_width
        self.n_cls = n_cls

        self.conv1 = nn.Conv2d(3, model_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 2 * model_width, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 4 * model_width, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 8 * model_width, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(8 * model_width * block.expansion)
        self.linear = nn.Linear(8 * model_width * block.expansion, 1 if n_cls == 2 else n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes, model_width=64):
    bn_flag = True
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=num_classes, model_width=model_width)


def PreActResNet34(num_classes, model_width=64):
    bn_flag = True
    return PreActResNet(PreActBlock, [3, 4, 6, 3], n_cls=num_classes, model_width=model_width)


def PreActResNet50(num_classes, model_width=64):
    bn_flag = True
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], n_cls=num_classes, model_width=model_width)
