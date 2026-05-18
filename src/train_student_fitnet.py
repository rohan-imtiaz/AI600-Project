# AI 600 Project 15: Knowledge Distillation (train_student_fitnet.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/train_student_fitnet.py
# Performs two-stage FitNets training on ResNet-20 using ResNet-110 as teacher.
# Stage 1: trains the student's first half to match teacher intermediate features (hint training).
# Stage 2: trains the full student end-to-end using Hinton KD from the Stage 1 initialization.
# Teacher checkpoint must already exist at checkpoints/teacher_resnet110.pth.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss, hint_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

# Convolutional regressor that projects student hint features to teacher hint channel size

class HintRegressor(nn.Module):

    def __init__(self, student_channels, teacher_channels):
        super(HintRegressor, self).__init__()

        # 1x1 conv followed by batch norm and ReLU to project channel dimensions
        self.regressor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.regressor(x)

# Stage 1: pre-trains the student's early layers to match teacher intermediate representations

def train_stage1_hints(student_model, teacher_model, training_loader, device, number_of_stage1_epochs, results_directory):

    print("Stage 1: Hint-based pre-training")

    student_hint_channels = 32 # ResNet-20 stage2 output channels
    teacher_hint_channels = 32 # ResNet-110 stage2 output channels

    regressor = HintRegressor(student_hint_channels, teacher_hint_channels).to(device)

    # Collecting only the student parameters up to and including stage2 for Stage 1 optimization
    student_early_parameters = (
        list(student_model.conv1.parameters()) +
        list(student_model.bn1.parameters())   +
        list(student_model.stage1.parameters()) +
        list(student_model.stage2.parameters())
    )

    optimizer = optim.SGD(
        student_early_parameters + list(regressor.parameters()),
        lr           = 0.01,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    log_file_path = os.path.join(results_directory, "fitnet_stage1_log.csv")
    initialize_csv_log(log_file_path, ["epoch", "hint_loss"])

    for epoch in range(1, number_of_stage1_epochs + 1):

        student_model.train()
        regressor.train()
        total_hint_loss = 0.0

        progress_bar = tqdm(training_loader, desc = f"Stage 1 Epoch {epoch}/{number_of_stage1_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)

            with torch.no_grad(): # Getting teacher hint features with no gradient tracking
                teacher_hint_features = teacher_model.get_hint_features(images)

            optimizer.zero_grad()
            student_hint_features = student_model.get_hint_features(images)

            loss = hint_loss(student_hint_features, teacher_hint_features, regressor)
            loss.backward()
            optimizer.step()

            total_hint_loss = total_hint_loss + loss.item()
            progress_bar.set_postfix({"hint_loss": loss.item()})

        average_hint_loss = total_hint_loss / len(training_loader)
        print(f"Stage 1 Epoch {epoch}: Hint Loss = {average_hint_loss:.6f}")
        append_csv_row(log_file_path, [epoch, average_hint_loss])

    return regressor

# Stage 2: full KD training starting from the hint-pretrained student weights

def train_stage2_kd(student_model, teacher_model, training_loader, test_loader,
                     device, number_of_stage2_epochs, temperature, alpha,
                     checkpoint_directory, results_directory):

    print("Stage 2: Full KD training from hint-pretrained initialization")

    optimizer = optim.SGD(
        student_model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    log_file_path = os.path.join(results_directory, "student_fitnet_log.csv")
    initialize_csv_log(log_file_path, ["epoch", "train_loss", "train_acc", "test_acc"])

    best_test_accuracy = 0.0

    for epoch in range(1, number_of_stage2_epochs + 1):

        student_model.train()
        total_loss          = 0.0
        correct_predictions = 0
        total_samples       = 0

        progress_bar = tqdm(training_loader, desc = f"Stage 2 Epoch {epoch}/{number_of_stage2_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad(): # Getting teacher soft labels with no gradient tracking
                teacher_logits, _ = teacher_model(images)

            optimizer.zero_grad()
            student_logits, _ = student_model(images)

            loss = knowledge_distillation_loss(
                student_logits, teacher_logits, labels, temperature, alpha
            )

            loss.backward()
            optimizer.step()

            total_loss          = total_loss + loss.item()
            predicted_classes   = student_logits.argmax(dim = 1)
            correct_predictions = correct_predictions + (predicted_classes == labels).sum().item()
            total_samples       = total_samples + labels.size(0)

        scheduler.step()

        train_accuracy = 100.0 * correct_predictions / total_samples
        average_loss   = total_loss / len(training_loader)
        test_accuracy  = compute_accuracy(student_model, test_loader, device)

        print(f"Epoch {epoch}: Loss = {average_loss:.4f}  Train Acc = {train_accuracy:.2f}%  Test Acc = {test_accuracy:.2f}%")

        append_csv_row(log_file_path, [epoch, average_loss, train_accuracy, test_accuracy])

        if test_accuracy > best_test_accuracy: # Saving best FitNet checkpoint found so far
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, "student_fitnet.pth"))

    print(f"Best FitNet student test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

# Entry point: runs Stage 1 then Stage 2 in sequence

def train_student_fitnet(temperature = 4.0, alpha = 0.9):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    # Loading and freezing the pretrained teacher
    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, "teacher_resnet110.pth")
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    student_model = build_resnet20().to(device)

    # Stage 1: 50 epochs of hint-based pre-training
    train_stage1_hints(
        student_model, teacher_model, training_loader,
        device,
        number_of_stage1_epochs = 50,
        results_directory       = results_directory
    )

    # Stage 2: 200 epochs of full KD from the hint-initialized student
    best_accuracy = train_stage2_kd(
        student_model, teacher_model, training_loader, test_loader,
        device,
        number_of_stage2_epochs = 200,
        temperature             = temperature,
        alpha                   = alpha,
        checkpoint_directory    = checkpoint_directory,
        results_directory       = results_directory
    )

    return best_accuracy

if __name__ == "__main__":
    train_student_fitnet(temperature = 4.0, alpha = 0.9)