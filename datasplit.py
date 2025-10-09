import argparse
from ast import arg
from concurrent.futures import ThreadPoolExecutor
import shutil
import zipfile
import os
import random

from tqdm import tqdm

from utils.Utilities import YAML_Reader
random.seed(42)

def Unzip_File(path):
    extract_to = "./datasets"

    os.makedirs(extract_to, exist_ok=True) 

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("Extract {0} to {1}".format(path, extract_to))

def copy_file_task(args):
    src, dst = args
    if not os.path.exists(dst):
        shutil.copy(src, dst)

def Split_Data(source_folder, train_folder, test_folder, split_ratio=0.6, selection_ratio=0.3):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    class_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    for folder in tqdm(class_folders, desc="Splitting dataset", total=len(class_folders)):
        folder_path = os.path.join(source_folder, folder)
        all_files = sorted([
            file for file in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, file))
        ])

        # Shuffle and select only a portion of the data
        random.shuffle(all_files)
        selection_count = int(len(all_files) * selection_ratio)
        selected_files = all_files[:selection_count]

        # Train/test split
        split_index = int(len(selected_files) * split_ratio)
        train_files = selected_files[:split_index]
        test_files = selected_files[split_index:]

        # Output directories
        sub_train_folder = os.path.join(train_folder, folder)
        sub_test_folder = os.path.join(test_folder, folder)
        os.makedirs(sub_train_folder, exist_ok=True)
        os.makedirs(sub_test_folder, exist_ok=True)

        # Prepare tasks for multithreaded copy
        tasks = []
        for file in train_files:
            src = os.path.join(folder_path, file)
            dst = os.path.join(sub_train_folder, file)
            tasks.append((src, dst))
        for file in test_files:
            src = os.path.join(folder_path, file)
            dst = os.path.join(sub_test_folder, file)
            tasks.append((src, dst))

        # Multithreaded copying (adjust workers as needed)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(copy_file_task, tasks))

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
def main():
    config = parse_args()
    Unzip_File(path=config["DATASET"]["ZIP_FOLDER"])
    Split_Data(source_folder=config["DATASET"]["ROOT_FOLDER"],
               train_folder=config["DATASET"]["TRAIN_FOLDER"],
               test_folder=config["DATASET"]["TEST_FOLDER"],
               selection_ratio=config["DATASET"]["SELECTION_RATIO"],
               split_ratio=config["DATASET"]["SPLIT_RATIO"]
               )
    
if __name__ == '__main__':
    main()