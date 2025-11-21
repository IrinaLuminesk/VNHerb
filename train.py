import argparse
import os
from typing import Sequence
from sympy import Float
from tqdm import tqdm
from Aug.BatchWiseAug import BatchWiseAug
from Metrics.MetricCal import MetricCal
from learning_rate import PiecewiseScheduler
from model import Model
from utils.DatasetLoader import DatasetLoader
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric, Saving_Metric2, YAML_Reader, get_mean_std

import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss.cross_entropy import SoftTargetCrossEntropy

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

def set_seed(seed=42):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch: int, end_epoch: int, batchWiseAug, model, loader, criterion, optimizer, device, num_classes):
    model.train()
    metrics = MetricCal(num_classes=num_classes)
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = batchWiseAug(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        metrics.update(loss=loss, outputs=outputs, targets=targets, type="soft")
    return metrics

def validate(epoch, end_epoch, model, loader, criterion, device, num_classes):
    model.eval()
    metrics = MetricCal(num_classes=num_classes)
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metrics.update(loss=loss, outputs=outputs, targets=targets, type="hard")
    return metrics

def main():
    config = parse_args()


    #Data parameters
    root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    test_path = config["DATASET"]["TEST_FOLDER"]
    CLASSES = sorted([i for i in os.listdir(root_path)])
    mean: Sequence[float] = config["TRAIN"]["DATA"]["MEAN"]
    std: Sequence[float] = config["TRAIN"]["DATA"]["STD"]
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
    
    set_seed()
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean: Sequence[float]; std: Sequence[float] = get_mean_std(train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DatasetLoader(path=train_path, batch_size=batch_size, std=std, mean=mean, img_size=img_size)
    test_data = DatasetLoader(path=test_path, batch_size=batch_size, std=std, mean=mean, img_size=img_size)

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    batchWiseAug = BatchWiseAug(config=config, num_classes=len(CLASSES))

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
        train_metrics = train(epoch, 
                                end_epoch, 
                                batchWiseAug=batchWiseAug,
                                model=model, 
                                loader=training_loader, 
                                criterion=train_criterion, 
                                optimizer=optimizer, 
                                device=device,
                                num_classes=len(CLASSES))
        train_loss, train_acc = train_metrics.avg_loss, train_metrics.avg_accuracy
        scheduler.step()
        print()
        val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device, num_classes=len(CLASSES))
        val_loss, val_acc = val_metrics.avg_loss, val_metrics.avg_accuracy
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}, Training Acc: {3}%".
            format(epoch, end_epoch, train_loss, round(train_acc * 100.0, 2)))
        print("Epoch [{0}/{1}]: Validation loss: {2}, Validation Acc: {3}%".
            format(epoch, end_epoch, val_loss, round(val_acc * 100.0, 2)))
        if val_acc > best_acc:
            if save_best == True:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
                    format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
                    format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
            best_acc = val_acc
        if save_metrics:
            Saving_Metric2(epoch=epoch, 
                           train_loss=train_loss,
                           train_acc=train_acc,
                           train_precision=train_metrics.precision_macro,
                           train_recall=train_metrics.recall_macro,
                           train_f1=train_metrics.f1_macro, 
                           val_loss=val_loss,
                           val_acc=val_acc,
                           val_precision=val_metrics.precision_macro,
                           val_recall=val_metrics.recall_macro,
                           val_f1=val_metrics.f1_macro, 
                           path=metrics_path)
        print()

    
if __name__ == '__main__':
    main()
