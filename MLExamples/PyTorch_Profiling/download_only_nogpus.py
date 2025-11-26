#!/usr/bin/env python3

"""Mini extract from train_cifar_100.py, suitable for only downloading the data on a machine without GPUs"""

__author__ = "Luka Stanisic, AMD"
__email__  = "luka.stanisic@amd.com"

import argparse
import torch

# Dataset transforms:
import torchvision
from torchvision.transforms import v2


if __name__ == "__main__":

    # Create an argument parser:
    parser = argparse.ArgumentParser()

    # Data arguments:
    parser.add_argument("--data-path", "-dp",
                        help="Top level data storage",
                        type=str,
                        default="data/")

    args = parser.parse_args()

    training_data = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=True,
        download=True,
        transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop(size=32, scale=[0.85,1.0], antialias=False),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    )

    val_data = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=False,
        download=True,
        transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
        ])
    )
