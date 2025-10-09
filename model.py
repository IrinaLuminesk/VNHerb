import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(model_type, num_classes):
        match model_type:
            case "Resnet50":
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)

                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                return model

class Model(nn.Module):
    def __init__(self, num_classes, model_type, pretrained=True):
        super().__init__()

        self.model = build_model(model_type, num_classes)

    def forward(self, x):
        model = self.model(x)
        return model
    