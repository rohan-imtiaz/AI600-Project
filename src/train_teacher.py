# AI 600 Project 15: Knowledge Distillation (train_teacher.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/train_teacher.py
# Trains ResNet-110 on CIFAR-100 and saves the best checkpoint to checkpoints/.
# Intermediate checkpoints are also saved at epochs 50, 80, 100, 130, 150 for the early stopping ablation.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110
from src.utils import compute_accuracy, save_checkpoint, initialize_csv_log, append_csv_row, get_device

def train_teacher():

    device = get_device()

    data_directory        = "./data"
    checkpoint_directory  = "./checkpoints"
    results_directory     = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model = build_resnet110().to(device)

    # SGD with momentum is the standard optimizer for CIFAR ResNet training
    optimizer = optim.SGD(
        teacher_model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    # Learning rate decayed by 0.1 at epochs 100 and 150
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    loss_function    = nn.CrossEntropyLoss()
    number_of_epochs = 200
    log_file_path    = os.path.join(results_directory, "teacher_log.csv")

    initialize_csv_log(log_file_path, ["epoch", "train_loss", "train_acc", "test_acc"])

    # Epochs at which to save intermediate checkpoints for the early stopping ablation
    intermediate_checkpoint_epochs = [50, 80, 100, 130, 150]

    best_test_accuracy = 0.0

    for epoch in range(1, number_of_epochs + 1):

        teacher_model.train()
        total_loss          = 0.0
        correct_predictions = 0
        total_samples       = 0

        progress_bar = tqdm(training_loader, desc = f"Teacher Epoch {epoch}/{number_of_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = teacher_model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss          = total_loss + loss.item()
            predicted_classes   = logits.argmax(dim = 1)
            correct_predictions = correct_predictions + (predicted_classes == labels).sum().item()
            total_samples       = total_samples + labels.size(0)

            progress_bar.set_postfix({"loss": loss.item()})

        scheduler.step()

        train_accuracy = 100.0 * correct_predictions / total_samples
        average_loss   = total_loss / len(training_loader)
        test_accuracy  = compute_accuracy(teacher_model, test_loader, device)

        print(f"Epoch {epoch}: Loss = {average_loss:.4f}  Train Acc = {train_accuracy:.2f}%  Test Acc = {test_accuracy:.2f}%")

        append_csv_row(log_file_path, [epoch, average_loss, train_accuracy, test_accuracy])

        if test_accuracy > best_test_accuracy: # Saving the best checkpoint found so far
            best_test_accuracy = test_accuracy
            save_checkpoint(teacher_model, os.path.join(checkpoint_directory, "teacher_resnet110.pth"))

        if epoch in intermediate_checkpoint_epochs: # Saving intermediate snapshots for ablation study
            intermediate_name = f"teacher_epoch_{epoch}.pth"
            save_checkpoint(teacher_model, os.path.join(checkpoint_directory, intermediate_name))

    print(f"Best teacher test accuracy: {best_test_accuracy:.2f}%")

if __name__ == "__main__":
    train_teacher()