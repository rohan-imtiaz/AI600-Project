# AI 600 Project 15: Knowledge Distillation (train_student_kd.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/train_student_kd.py
# Trains ResNet-20 using Hinton et al. soft-label knowledge distillation from ResNet-110.
# Teacher checkpoint must already exist at checkpoints/teacher_resnet110.pth.
# The function is also called directly by the ablation scripts with different arguments.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

def train_student_kd(temperature = 4.0, alpha = 0.9, teacher_checkpoint = "teacher_resnet110.pth", run_name = "student_kd"):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    # Loading the pretrained teacher and freezing all its parameters
    teacher_model          = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, teacher_checkpoint)
    teacher_model          = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    student_model = build_resnet20().to(device)

    optimizer = optim.SGD(
        student_model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    number_of_epochs = 200
    log_file_path    = os.path.join(results_directory, f"{run_name}_log.csv")

    initialize_csv_log(log_file_path, ["epoch", "train_loss", "train_acc", "test_acc"])

    best_test_accuracy = 0.0

    print(f"KD Training - Temperature: {temperature}, Alpha: {alpha}")

    for epoch in range(1, number_of_epochs + 1):

        student_model.train()
        total_loss          = 0.0
        correct_predictions = 0
        total_samples       = 0

        progress_bar = tqdm(training_loader, desc = f"KD Epoch {epoch}/{number_of_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad(): # Getting teacher soft labels without computing gradients
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

        if test_accuracy > best_test_accuracy: # Saving best checkpoint found so far
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, f"{run_name}.pth"))

    print(f"Best student (KD) test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

if __name__ == "__main__":
    train_student_kd(temperature = 4.0, alpha = 0.9)