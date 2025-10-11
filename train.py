import argparse
import os
from tqdm import tqdm
from model import Model
from utils.Utilities import YAML_Reader, get_mean_std

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/default_config.yaml",
    help="Config file used to train the model (default: config/default_config.yaml)"
    )
    args = parser.parse_args()
    config = YAML_Reader(args.cfg)
    return config

def Get_Dataset(train_path, test_path, train_transform, test_transform, batch_size):
    training_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
    print("Total train image: {0}".format(len(training_dataset)))
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testing_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transform
    )
    print("Total test image: {0}".format(len(testing_dataset)))
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print()
    return training_loader, testing_loader

def Get_Transform(mean: list, std: list):
    training_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.RandomChoice([
            v2.RandomHorizontalFlip(p=1.0),
            v2.RandomVerticalFlip(p=1.0),
            v2.RandomRotation(degrees=15, interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomAffine(degrees=0, translate=(0.5, 0.1)),
            v2.RandomResizedCrop((256, 256), scale=(0.5, 1.1), ratio=(1.0, 1.0)),
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            v2.Lambda(lambda x: x),
            ]),
            v2.ToTensor(),
            v2.Normalize(
                mean=mean,
                std=std
            )
    ])

    testing_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToTensor(),
        v2.Normalize(
            mean=mean,
            std=std
        )
    ])

    return training_transform, testing_transform

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in tqdm(loader, total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    test_path = config["DATASET"]["TEST_FOLDER"]
    CLASSES = sorted([i for i in os.listdir(root_path)])
    
    
    mean = config["TRAIN"]["MEAN"]
    std = config["TRAIN"]["STD"]
    batch_size = config["TRAIN"]["BATCH_SIZE"]
    begin_epoch = config["TRAIN"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["END_EPOCH"]
    resume = config["TRAIN"]["RESUME"]
    if resume == True:
        begin_epoch = config["TRAIN"]["LAST_EPOCH"]
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean, std = get_mean_std(train_path)
    
    training_transform, testing_transform = Get_Transform(mean=mean, std=std)
    
    training_loader, testing_loader = Get_Dataset(train_path=train_path,
                                                   test_path=test_path, 
                                                   train_transform=training_transform, 
                                                   test_transform=testing_transform, 
                                                   batch_size=batch_size)

    model = Model(100, "Resnet50").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
    
    for epoch in range(begin_epoch, end_epoch):
        train_loss, train_acc = train(model, training_loader, criterion, optimizer, device=device)
        # val_loss, val_acc = validate(model, testing_loader, criterion)
        print("Epoch [{0}/{1}]: Training loss: {2}\tTraining Acc: {3}%".
            format(epoch, end_epoch, train_loss, round(train_acc, 2)))
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_epoch = epoch
        #     Save_Best(model, best_acc, best_epoch, path="/content/best.pth")
        # else:
        #     print("Model didn't improve from {0}%".format(round(best_acc, 2)))
        # # current_lr = optimizer.param_groups[0]['lr']
        # Save_Metrics(epoch, train_loss, train_acc, val_loss, val_acc, 0.0, path="/content/Hybrid_training.csv")
        # # cosine_scheduler.step()
        # Save_Checkpoint(epoch, model, optimizer, best_acc, best_epoch)
        # # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        
    
if __name__ == '__main__':
    main()