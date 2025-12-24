import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock

import re

from mi.nets import MIFCNet, MI1x1ConvNet

def resnet18(inplanes=BasicBlock, planes=[2, 2, 2, 2]):
    model = ResNetMI(inplanes, planes, True)
    return model

class ResNetMI(ResNet):

    def __init__(self, inplanes, planes, mutual_info_loss):
        super(ResNetMI, self).__init__(inplanes, planes)
        self.mutual_info_loss = mutual_info_loss
        local_units = 512
        mi_units = 512
        global_units = 512
        self.local_net = MI1x1ConvNet(local_units, mi_units)
        self.global_net = MIFCNet(global_units, mi_units)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = x

        features = self.local_net(features)

        vector = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        vector = self.global_net(vector)

        return features, vector
