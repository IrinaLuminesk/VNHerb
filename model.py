import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,\
    densenet201, DenseNet201_Weights,\
    vgg19, VGG19_Weights, \
    convnext_base, ConvNeXt_Base_Weights,\
    mobilenet_v2, MobileNet_V2_Weights, \
    swin_v2_b, Swin_V2_B_Weights

# def build_model(model_type: int, num_classes: int):
#         match model_type:
#             case 1: #Resnet50
#                 resnet_weights = ResNet50_Weights.DEFAULT
#                 model = resnet50(weights=resnet_weights)

#                 in_features = model.fc.in_features #2048
#                 fc = nn.Sequential(
#                     nn.Linear(in_features, 1024),
#                     nn.BatchNorm1d(1024),
#                     nn.ReLU(),
#                     nn.Dropout(0.4),
#                     nn.Linear(1024, num_classes),
#                 )
#                 model.fc = fc
#                 return model
#             case 2: #DenseNet201
#                 densenet_Weights = DenseNet201_Weights.DEFAULT
#                 model = densenet201(weights=densenet_Weights)

#                 in_features = model.classifier.in_features #1920
#                 fc = nn.Sequential(
#                     nn.Linear(in_features, 1024),
#                     nn.BatchNorm1d(1024),
#                     nn.ReLU(),
#                     nn.Dropout(0.4),
#                     nn.Linear(1024, num_classes),
#                 )
#                 model.classifier = fc

#                 return model
#             case 3: #VGG19
#                 vgg19_weights = VGG19_Weights.DEFAULT
#                 model = vgg19(weights=vgg19_weights)

#                 vgg19_classifier = list(model.classifier.children())[:6]
#                 in_features = model.classifier[6].in_features #4096

#                 model.classifier = nn.Sequential(
#                     *vgg19_classifier,
#                     nn.Linear(in_features, 2048, bias=True),
#                     nn.BatchNorm1d(2048),
#                     nn.ReLU(),
#                     nn.Dropout(0.4),
#                     nn.Linear(2048, 1024, bias=True),
#                     nn.BatchNorm1d(1024),
#                     nn.ReLU(),
#                     nn.Dropout(0.4),
#                     nn.Linear(1024, num_classes)
#                 )
#                 return model
#             case 4: #convnext base
#                 convnext_weight = ConvNeXt_Base_Weights.DEFAULT
#                 model = convnext_base(weights= convnext_weight)

#                 convnext_classifier = list(model.classifier.children())[:2]
#                 in_features = model.classifier[2].in_features

#                 model.classifier = nn.Sequential(
#                     *convnext_classifier,
#                     nn.Linear(in_features, num_classes, bias=True)
#                 )
#                 return model
#             case 5: #MobileNet
#                 mobilenetv2_weights = MobileNet_V2_Weights.DEFAULT
#                 model = mobilenet_v2(weights=mobilenetv2_weights)

#                 in_features = model.classifier[1].in_features #1280

#                 model.classifier = nn.Sequential(
#                     nn.Linear(in_features, 1024, bias=True),
#                     nn.BatchNorm1d(1024),
#                     nn.ReLU(),
#                     nn.Dropout(0.4),
#                     nn.Linear(1024, num_classes)
#                 )
#                 return model
#             case 6: #Swin transform
#                 swinv2Weight = Swin_V2_B_Weights.DEFAULT
#                 model = swin_v2_b(weights=swinv2Weight)

#                 in_features = model.head.in_features #1024
#                 model.head = nn.Linear(in_features, num_classes, bias=True)
#                 return model
            

class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self.build_model() 
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)

                in_features = model.fc.in_features #2048
                fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )
                model.fc = fc
                return model
            case 2: #DenseNet201
                densenet_Weights = DenseNet201_Weights.DEFAULT
                model = densenet201(weights=densenet_Weights)

                in_features = model.classifier.in_features #1920
                fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )
                model.classifier = fc

                return model
            case 3: #VGG19
                vgg19_weights = VGG19_Weights.DEFAULT
                model = vgg19(weights=vgg19_weights)

                vgg19_classifier = list(model.classifier.children())[:6]
                in_features = model.classifier[6].in_features #4096

                model.classifier = nn.Sequential(
                    *vgg19_classifier,
                    nn.Linear(in_features, 2048, bias=True),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(2048, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                return model
            case 4: #convnext base
                convnext_weight = ConvNeXt_Base_Weights.DEFAULT
                model = convnext_base(weights= convnext_weight)

                convnext_classifier = list(model.classifier.children())[:2]
                in_features = model.classifier[2].in_features

                model.classifier = nn.Sequential(
                    *convnext_classifier,
                    nn.Linear(in_features, self.num_classes, bias=True)
                )
                return model
            case 5: #MobileNet
                mobilenetv2_weights = MobileNet_V2_Weights.DEFAULT
                model = mobilenet_v2(weights=mobilenetv2_weights)

                in_features = model.classifier[1].in_features #1280

                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                return model
            case 6: #Swin transform
                swinv2Weight = Swin_V2_B_Weights.DEFAULT
                model = swin_v2_b(weights=swinv2Weight)

                in_features = model.head.in_features #1024
                model.head = nn.Linear(in_features, self.num_classes, bias=True)
                return model
    def forward(self, x):
        return self.model(x)
    