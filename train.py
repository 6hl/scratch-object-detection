"""File used to train object detection models from scratch
Use:
    python3 train.py --model rcnn --epochs 1 --data dataset.yaml
"""

import argparse

from zoo.utils.parse_yaml import parse_yaml
from zoo.models.init_model import init_model

parser = argparse.ArgumentParser()
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
required.add_argument("--model", choices=["rcnn"], required=True, help="Select model for training")
required.add_argument("--epochs", type=int, required=True, help="Enter training epochs")
required.add_argument("--data", type=str, required=True, help="Enter yaml path location for dataset")
optional.add_argument("--batch-size", type=int, default=16, help="total bactchsize for gpus")
optional.add_argument("--force-data", action="store_true", help="force generate new data")

def run(args):
    args.data = parse_yaml(args.data)
    init_model(vars(args))


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
