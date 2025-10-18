import argparse
import os
from tqdm import tqdm
import yaml
from learning_rate import PiecewiseScheduler
from model import Model
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric, YAML_Modify, YAML_Reader, get_mean_std
from ruamel.yaml import YAML

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

def Get_Dataset(train_path, test_path, train_transform, test_transform, batch_size, use_ddp):
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
    training_sampler, testing_sampler = None, None
    if use_ddp:
        training_sampler = DistributedSampler(training_dataset)
        training_loader = DataLoader(training_dataset, 
                                     batch_size=batch_size, 
                                     sampler=training_sampler,
                                     shuffle=True)
        
        testing_sampler = DistributedSampler(testing_dataset, shuffle=False)
        testing_loader = DataLoader(testing_dataset, 
                                     batch_size=batch_size, 
                                     sampler=testing_sampler,
                                     shuffle=False)
    else:
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
    return training_loader, training_sampler, testing_loader, testing_sampler

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
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=mean,
                std=std
            )
    ])

    testing_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=mean,
            std=std
        )
    ])

    return training_transform, testing_transform

def train(epoch, end_epoch, model, loader, criterion, optimizer, device, use_ddp, sampler):
    model.train()
    if use_ddp:
        sampler.set_epoch(epoch) 
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
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

    if use_ddp:
        # convert to tensors
        total_loss_tensor = torch.tensor(total_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)

        # sum across all GPUs
        torch.distributed.all_reduce(total_loss_tensor)
        torch.distributed.all_reduce(correct_tensor)
        torch.distributed.all_reduce(total_tensor)

        # compute global averages
        avg_loss = total_loss_tensor.item() / total_tensor.item()
        accuracy = 100. * correct_tensor.item() / total_tensor.item()
    else:
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(epoch, end_epoch, model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

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

    yaml_o = YAML()
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean, std = get_mean_std(train_path)

    num_gpus = torch.cuda.device_count()
    use_ddp = num_gpus > 1
    if use_ddp:
        # local_rank = dist.get_rank()
        # torch.cuda.set_device(local_rank)
        # device = torch.device(f"cuda:{local_rank}")
        # dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"Using DDP with {num_gpus} GPUs. Local rank: {local_rank}")
        # torchrun --nproc_per_node=NUM_GPUS train.py
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        print(f"Single GPU or CPU. Using device: {device}")
        # python train.py
    
    training_transform, testing_transform = Get_Transform(mean=mean, std=std)
    
    training_loader, training_sampler, testing_loader, testing_sampler = Get_Dataset(train_path=train_path,
                                                   test_path=test_path, 
                                                   train_transform=training_transform, 
                                                   test_transform=testing_transform, 
                                                   batch_size=batch_size,
                                                   use_ddp=use_ddp)

    model = Model(len(CLASSES), model_type).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
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
    best_epoch = 0

    if resume == True:
        begin_epoch = Loading_Checkpoint(path="/checkpoint.pth",
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         use_ddp=use_ddp,
                                         map_location=local_rank)
        best_acc = Get_Max_Acc(metrics_path)
        if use_ddp:
            dist.barrier()

    # rank = dist.get_rank() if torch.distributed.is_initialized() else 0
    for epoch in range(begin_epoch, end_epoch):
        if use_ddp:
            training_sampler.set_epoch(epoch)
        train_loss, train_acc = train(epoch, 
                                      end_epoch, 
                                      model, 
                                      training_loader, 
                                      criterion, 
                                      optimizer, 
                                      device=device, 
                                      use_ddp=use_ddp, 
                                      sampler=training_sampler)
        scheduler.step()
        print()
        val_loss, val_acc = validate(epoch, end_epoch, model, testing_loader, criterion, device)
        print()

        if local_rank == 0:
            if save_checkpoint == True:
                Saving_Checkpoint(epoch=epoch, 
                                model=model, 
                                optimizer=optimizer, 
                                scheduler=scheduler,
                                last_epoch=epoch, 
                                path=checkpoint_path, 
                                use_ddp=use_ddp)
                # YAML_Modify(yaml_o=yaml_o,
                #             path="config/default_config.yaml",
                #             key=["TRAIN", "TRAIN_PARA", "LAST_EPOCH"],
                #             value=epoch)

            print("Epoch [{0}/{1}]: Training loss: {2}, Training Acc: {3}%".
                format(epoch, end_epoch, train_loss, round(train_acc, 2)))
            print("Epoch [{0}/{1}]: Validation loss: {2}, Validation Acc: {3}%".
                format(epoch, end_epoch, val_loss, round(val_acc, 2)))
            if val_acc > best_acc:
                if save_best == True:
                    print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
                        format(round(best_acc, 2), round(val_acc, 2),  epoch))
                    Saving_Best(model, best_path, use_ddp=use_ddp)
                else:
                    print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
                        format(round(best_acc, 2), round(val_acc, 2),  epoch))
                best_acc = val_acc
                best_epoch = epoch
        if save_metrics == True:
            Saving_Metric(epoch=epoch, 
                          train_acc=train_acc, 
                          train_loss=train_loss, 
                          val_acc=val_acc, 
                          val_loss=val_loss, 
                          path=metrics_path)
        print()

    
if __name__ == '__main__':
    main()