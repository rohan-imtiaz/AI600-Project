# AI 600 Project 15: Knowledge Distillation (dataset.py)

# INSTRUCTIONS:
# This file is not run directly.
# It is imported by all training scripts to provide CIFAR-100 data loaders.

import torch
import torchvision
import torchvision.transforms as transforms

# Building training and test data loaders for CIFAR-100

def get_cifar100_loaders(data_directory, batch_size = 128, num_workers = 2):

    # CIFAR-100 channel-wise mean and std computed over the training set
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    # Training transform includes random crop and flip for data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test transform only normalizes, no augmentation applied
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    training_dataset = torchvision.datasets.CIFAR100(
        root = data_directory,
        train = True,
        download = True,
        transform = train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root = data_directory,
        train = False,
        download = True,
        transform = test_transform
    )

    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    return training_loader, test_loader