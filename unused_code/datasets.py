import os
import time
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

def get_train_valid_loader(train_dir,
                            valid_dir,
                            train_batch_size,
                            val_batch_size,
                            train_transform,
                            valid_transform,
                            num_workers=4,
                            pin_memory=False):
    """
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    # load the dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

    classes = train_dataset.classes
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)

    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, train_data_size, valid_data_size, classes