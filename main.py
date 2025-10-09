import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument("--train", type=str, required=True, help="Training data path")
    parser.add_argument("--test", type=str, required=True, help="Testing data path")

    return parser.parse_args()
def main():
    args = parse_args()
    
if __name__ == '__main__':
    main()