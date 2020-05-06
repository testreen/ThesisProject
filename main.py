import os
import matplotlib.image as mpimg
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import torchvision
from torch.utils import data
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import time

from parseData import parseData
from efficientnet_pytorch import EfficientNet

from visualize_model import visualize_model
from train_valid_split import train_valid_split
from run_model import run_model
from CellDataset import CellDataset

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch EfficientNet Training')
parser.add_argument('--data', metavar='DIR', default="",
                    help='path to KI-Dataset folder')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: efficientnet-b0)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('-val', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--feature_extract', dest='feature_extract',
                    action='store_true',
                    help="Train only last layer (otherwise full model)")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=32, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')

# Static config
num_classes = 5
class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']
shuffle = True
k = 5 # Cross-validation splits

def lambdaTransform(image):
    return image * 2.0 - 1.0

def main():
    args = parser.parse_args()
    print(torch.version.cuda)
    a = torch.cuda.FloatTensor([1.])
    print(a)

    mp.set_start_method('spawn')

    # Normalize using dataset mean + std or advprop settings
    if args.advprop:
        normalize = transforms.Lambda(lambdaTransform)
    else:
        normalize = transforms.Normalize(mean=[0.72482513, 0.59128926, 0.76370454],
                                         std=[0.18745105, 0.2514997,  0.15264913])

    image_size = args.image_size
    print('Using image size', image_size)

    train_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size+2, interpolation=PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Load and split datasets and convert to tensor
    # Test images from different slices than train
    images, labels = parseData(basePath=args.data)

    train_images = images[:min(20390, len(images)-20)]
    train_labels = labels[:min(20390, len(images)-20)]
    test_images = images[min(20390, len(images)-20):]
    test_labels = labels[min(20390, len(images)-20):]

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transform to torch tensor
    tensor_test_x = torch.tensor(test_images, dtype=torch.float32, device=device)
    tensor_test_y = torch.tensor(test_labels, dtype=torch.long, device=device)
    tensor_test_x = tensor_test_x.permute(0, 3, 1, 2)

    split = 0

    for train, val in skf.split(train_images, train_labels):
        tensor_train_x = torch.tensor([train_images[i] for i in train], dtype=torch.float32, device=device)
        tensor_val_x = torch.tensor([train_images[i] for i in val], dtype=torch.float32, device=device)
        tensor_train_y = torch.tensor([train_labels[i] for i in train], dtype=torch.long, device=device)
        tensor_val_y = torch.tensor([train_labels[i] for i in val], dtype=torch.long, device=device)

        # Order array dimensions to pytorch standard
        tensor_train_x = tensor_train_x.permute(0, 3, 1, 2)
        tensor_val_x = tensor_val_x.permute(0, 3, 1, 2)


        train_dataset = CellDataset(tensors=(tensor_train_x, tensor_train_y),
                                    transform=train_tsfm)
        val_dataset = CellDataset(tensors=(tensor_val_x, tensor_val_y),
                                  transform=val_tsfm)
        test_dataset = CellDataset(tensors=(tensor_test_x, tensor_test_y),
                                   transform=val_tsfm)

        # Sizes of datasets
        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)
        test_dataset_size = len(test_dataset)
        print("train size: {}".format(train_dataset_size))
        print("val size: {}".format(val_dataset_size))
        print("test size: {}".format(test_dataset_size))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        model = run_model(loaders, split, args)
        split += 1

    # View results of model
    # visualize_model(model, my_dataloader)
    # plt.show()

    # View single image
    # crop = Image.fromarray(images[5814])
    # crop.show()
    # print(labels[5814])


if __name__ == '__main__':
    main()
