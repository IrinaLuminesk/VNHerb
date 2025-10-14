from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
from tqdm import tqdm
import yaml
import numpy as np
import os

def YAML_Reader(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def read_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.mean(axis=(0, 1)), img.std(axis=(0, 1)), 1
    except:
        return np.zeros(3), np.zeros(3), 0

def get_mean_std(path, max_workers=4):
    image_paths = []
    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, img_file))

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mean, std, count in tqdm(executor.map(read_img, image_paths), total=len(image_paths), desc="Computing mean/std"):
            total_mean += mean
            total_std += std
            total_count += count

    if total_count == 0:
        raise ValueError("No valid images found.")

    mean = total_mean / total_count
    std = total_std / total_count
    return mean, std

def Saving_Checkpoint(epoch, model, optimizer, scheduler, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)
def Saving_Best(model, path):
    torch.save(model.state_dict(), path)
