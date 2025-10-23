import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

dataset = datasets.ImageFolder(
        root="./datasets/Cifar10",
        transform=v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
        ])
)
print("Total train image: {0}".format(len(dataset)))

training_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)
print()

cutmix = v2.CutMix(num_classes=10, alpha=2.0)


images, labels = next(iter(training_loader))

# Apply CutMix
cutmix_images, cutmix_labels = cutmix(images, labels)