import argparse
import os
from tqdm import tqdm
import yaml
from RingMix import RingMix
from learning_rate import PiecewiseScheduler
from model import Model
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric, YAML_Reader, get_mean_std

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from timm.loss import SoftTargetCrossEntropy

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
    testing_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transform
    )
    print("Total test image: {0}".format(len(testing_dataset)))
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testing_loader = DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print()
    return training_loader, testing_loader

def Get_Transform(mean: list, std: list, img_size):
    training_transform = v2.Compose([
        v2.Resize(img_size),
        v2.RandomChoice([
            # v2.RandomHorizontalFlip(p=1.0),
            # v2.RandomVerticalFlip(p=1.0),
            # v2.RandomRotation(degrees=15, interpolation=v2.InterpolationMode.BILINEAR),
            # v2.RandomAffine(degrees=0, translate=(0.5, 0.1)),
            # v2.RandomResizedCrop((256, 256), scale=(0.5, 1.1), ratio=(1.0, 1.0)),
            # v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            # v2.RandomResizedCrop(size=size),
            v2.RandomHorizontalFlip(p=1),
            v2.RandomVerticalFlip(p=1),
            v2.Compose([
                v2.Pad((10, 20)),
                v2.Resize(img_size)
            ]),
            v2.Compose([
                v2.RandomZoomOut(p=1, side_range=(1, 1.5)),
                v2.Resize(img_size)
            ]),
            v2.RandomRotation(degrees=(-180, 180)),
            v2.RandomAffine(degrees=(-180, 180), translate=(0.1, 0.3), scale=(0.5, 1.75)),
            v2.RandomPerspective(p=1),
            v2.ElasticTransform(alpha=120),
            v2.ColorJitter(brightness=(1,2), contrast=(1,2)),
            v2.RandomPhotometricDistort(brightness=(1,2), contrast=(1,2), p=1),
            v2.RandomChannelPermutation(),
            v2.RandomGrayscale(p=1),
            v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 4.75)),
            v2.RandomInvert(p=1),
            v2.Lambda(lambda x: x),
            ]),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=mean,
                std=std
            )
    ])

    testing_transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=mean,
            std=std
        )
    ])

    return training_transform, testing_transform

def train(epoch: int, end_epoch: int, NUM_CLASSES: int, model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
        
        cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=2.0)
        mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=2.0)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=0.5)
        # ringmix = RingMix(patch_size=16, num_classes=10, p=0.5)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = cutmix_or_mixup(inputs, targets)
        # inputs, targets = ringmix(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        # accuracy = 100. * correct / total
    return avg_loss, 0

def validate(epoch, end_epoch, model, loader, criterion, device):
    model.eval()
    total_loss, correct_top1, correct_top5, total = 0, 0, 0
   
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total += targets.size(0)

            # Loss
            total_loss += loss.item() * inputs.size(0)
            # Accuracy
            #Top 1
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(targets).sum().item()
            #Top 5
            _, predicted = outputs.topk(5, 1, True, True)  # top 5 predicted class indices
            correct_top5 = predicted.eq(targets.view(-1, 1).expand_as(predicted)).sum().item()

    avg_loss = total_loss / total
    accuracy_top1 = 100. * correct_top1 / total
    accuracy_top5 = 100. * correct_top5 / total
    return avg_loss, accuracy_top1, accuracy_top5

def main():
    config = parse_args()


    #Data parameters
    root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    test_path = config["DATASET"]["TEST_FOLDER"]
    CLASSES = sorted([i for i in os.listdir(root_path)])
    mean = config["TRAIN"]["DATA"]["MEAN"]
    std = config["TRAIN"]["DATA"]["STD"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    

    #Training parameters
    img_size = config["TRAIN"]["DATA"]["IMAGE_SIZE"]
    begin_epoch = config["TRAIN"]["TRAIN_PARA"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["TRAIN_PARA"]["END_EPOCH"]
    resume = config["TRAIN"]["TRAIN_PARA"]["RESUME"]
    model_type = int(config["TRAIN"]["TRAIN_PARA"]["MODEL_TYPE"])

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean, std = get_mean_std(train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_transform, testing_transform = Get_Transform(mean=mean, std=std, img_size=img_size)
    
    training_loader, testing_loader = Get_Dataset(train_path=train_path,
                                                   test_path=test_path, 
                                                   train_transform=training_transform, 
                                                   test_transform=testing_transform, 
                                                   batch_size=batch_size)

    model = Model(len(CLASSES), model_type).to(device)

    eval_criterion = nn.CrossEntropyLoss()
    train_criterion = SoftTargetCrossEntropy()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    lr_schedule = PiecewiseScheduler(
        start_lr=0.0001,
        max_lr=0.0005,
        min_lr=0.0001,
        rampup_epochs=10,
        sustain_epochs=5,
        exp_decay=0.8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    best_acc = 0

    if resume == True:
        begin_epoch = Loading_Checkpoint(path=checkpoint_path,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         device=device)
        best_acc = Get_Max_Acc(metrics_path)

    for epoch in range(begin_epoch, end_epoch):
        train_loss, train_acc = train(epoch, 
                                      end_epoch, 
                                      NUM_CLASSES=CLASSES,
                                      model=model, 
                                      training_loader=training_loader, 
                                      train_criterion=train_criterion, 
                                      optimizer=optimizer, 
                                      device=device)
        scheduler.step()
        print()
        val_loss, top1_val_acc, top5_val_acc = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device)
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}, Training Acc: {3}%".
            format(epoch, end_epoch, train_loss, round(train_acc, 2)))
        print("Epoch [{0}/{1}]: Validation loss: {2}, Validation Acc: {3}%".
            format(epoch, end_epoch, val_loss, round(top1_val_acc, 2)))
        if top1_val_acc > best_acc:
            if save_best == True:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
                    format(round(best_acc, 2), round(top1_val_acc, 2),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
                    format(round(best_acc, 2), round(top1_val_acc, 2),  epoch))
            best_acc = top1_val_acc
        if save_metrics:
            Saving_Metric(epoch=epoch, 
                          train_acc=train_acc, 
                          train_loss=train_loss, 
                          top1_val_acc=top1_val_acc,
                          top5_val_acc=top5_val_acc, 
                          val_loss=val_loss, 
                          path=metrics_path)
        print()

    
if __name__ == '__main__':
    main()