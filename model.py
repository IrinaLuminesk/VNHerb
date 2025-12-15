import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,\
    densenet201, DenseNet201_Weights,\
    vgg16, VGG16_Weights, \
    convnext_base, ConvNeXt_Base_Weights,\
    mobilenet_v2, MobileNet_V2_Weights, \
    swin_v2_b, Swin_V2_B_Weights, \
    inception_v3, Inception_V3_Weights, \
    efficientnet_b4, EfficientNet_B4_Weights
import timm         

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
            case 2: #VGG16
                vgg16_weights = VGG16_Weights.DEFAULT
                model = vgg16(weights=vgg16_weights)

                vgg16_classifier = list(model.classifier.children())[:6]
                in_features = model.classifier[6].in_features #4096

                model.classifier = nn.Sequential(
                    *vgg16_classifier,
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
            
            case 3: #Xception
                model = timm.create_model(
                    'xception65',
                    pretrained=True
                )

                in_features = model.get_classifier().in_features
                
                Xception_classifier = model.head
                Xception_classifier = list(Xception_classifier.children())[:2]

                model.head = nn.Sequential(
                    *Xception_classifier,
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on Xception65 architecture")
                return model
            
            case 4: #EfficientNetB4
                model = efficientnet_b4(weights=EfficientNet_B4_Weights)

                in_features = model.classifier[1].in_features #1792

                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on EfficientNetB4 architecture")
                return model
            
            case 5: #DenseNet201
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
                print("Training on DenseNet201 architecture")
                return model
            
            case 6: #MobileNet
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
                print("Training on MobileNetV2 architecture")
                return model

            case 7: #convnext base
                convnext_weight = ConvNeXt_Base_Weights.DEFAULT
                model = convnext_base(weights= convnext_weight)

                convnext_classifier = list(model.classifier.children())[:2]
                in_features = model.classifier[2].in_features #1024

                model.classifier = nn.Sequential(
                    *convnext_classifier,
                    nn.Linear(in_features, self.num_classes)
                )
                print("Training on convnextBase architecture")
                return model
            
            case 8: #Beit
                model = timm.create_model('beit_base_patch16_224', pretrained=True)
                in_features = model.head.in_features #768

                model.head = nn.Linear(in_features, self.num_classes)
                print("Training on Beit architecture")
                return model

            # case 6: #Swin transform
            #     swinv2Weight = Swin_V2_B_Weights.DEFAULT
            #     model = swin_v2_b(weights=swinv2Weight)

            #     in_features = model.head.in_features #1024
            #     model.head = nn.Linear(in_features, self.num_classes, bias=True)
            #     return model
    def forward(self, x):
        return self.model(x)
    