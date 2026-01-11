import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import ResNet

from utils.CBAM import ChannelAttention, SpatialAttention

class Bottleneck_custom(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_custom, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = self.build_model() 
    def build_model(self):
        model = ResNet(block=Bottleneck_custome, layers=[3, 4, 6, 3])
        
        resnet_weights = ResNet50_Weights.DEFAULT
        pretrained_state_dict = resnet_weights.get_state_dict() #State dùng để load vào cùng với CBAM

        # Load matching weights (partial load for CBAM)
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

        in_features = model.fc.in_features #2048
        fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, self.num_classes),
        )
        model.fc = fc
        print("Training on Resnet50 + CBAM architecture")
        return model
    def forward(self, x):
        return self.model(x)
