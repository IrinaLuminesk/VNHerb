import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, densenet201, DenseNet201_Weights, vgg19, VGG19_Weights, convnext_large, ConvNeXt_Large_Weights

def build_model(model_type, num_classes):
        match model_type:
            case "Resnet50":
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)

                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                return model
            case "DenseNet":
                densenet_Weights = DenseNet201_Weights.DEFAULT
                model = densenet201(weights=densenet_Weights)

                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)

                return model
            case "VGG19":
                vgg19_weights = VGG19_Weights.DEFAULT
                model = vgg19(weights=vgg19_weights)

                vgg19_classifier = list(model.classifier.children())[:6]
                in_features = model.classifier[6].in_features

                model.classifier = nn.Sequential(
                    *vgg19_classifier,
                    nn.Linear(in_features, num_classes, bias=True)
                )
                return model
            case "convnext":
                convnext_weight = ConvNeXt_Large_Weights.DEFAULT
                model = convnext_large(weights= convnext_weight)

                convnext_classifier = list(model.classifier.children())[:2]
                in_features = model.classifier[2].in_features

                model.classifier = nn.Sequential(
                    *convnext_classifier,
                    nn.Linear(in_features, num_classes, bias=True)
                )
                return model
            case "Swim":
                return 1
            

class Model(nn.Module):
    def __init__(self, num_classes, model_type, pretrained=True):
        super().__init__()

        self.model = build_model(model_type, num_classes)

    def forward(self, x):
        model = self.model(x)
        return model
    