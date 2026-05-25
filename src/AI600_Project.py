# %%
from google.colab import drive
drive.mount("/content/drive")

import os

project_path = "/content/drive/MyDrive/25280058_Project"

print("Folder exists:", os.path.exists(project_path))
print("Contents:", os.listdir(project_path))

# %%
import sys

sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

print("Path set.")

# %%
import os
import sys

# Confirming GPU is available
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Setting the working directory to the project folder on Drive
os.chdir("/content/drive/MyDrive/25280058_Project")
print("Working directory:", os.getcwd())

# Confirming src is importable
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")
print("src files:", os.listdir("src"))

# %%
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import tqdm as tqdm_module

# Monkey-patching tqdm so all existing scripts get notebook bars automatically
tqdm_module.tqdm = tqdm_notebook
print("tqdm patched for notebook display.")

# %%
# Reconfirming working directory is correct before each training run
import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110
from src.utils import compute_accuracy, save_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

def train_teacher():

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model = build_resnet110().to(device)

    optimizer = optim.SGD(
        teacher_model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    loss_function    = nn.CrossEntropyLoss()
    number_of_epochs = 200
    log_file_path    = os.path.join(results_directory, "teacher_log.csv")

    initialize_csv_log(log_file_path, ["epoch", "train_loss", "train_acc", "test_acc"])

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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(teacher_model, os.path.join(checkpoint_directory, "teacher_resnet110.pth"))

        if epoch in intermediate_checkpoint_epochs:
            intermediate_name = f"teacher_epoch_{epoch}.pth"
            save_checkpoint(teacher_model, os.path.join(checkpoint_directory, intermediate_name))

    print(f"Best teacher test accuracy: {best_test_accuracy:.2f}%")

train_teacher()

# %%
# Cell 6 — Train student from scratch (baseline)

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet20
from src.utils import compute_accuracy, save_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

def train_student_scratch():

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    student_model = build_resnet20().to(device)

    optimizer = optim.SGD(
        student_model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    loss_function    = nn.CrossEntropyLoss()
    number_of_epochs = 200
    log_file_path    = os.path.join(results_directory, "student_scratch_log.csv")

    initialize_csv_log(log_file_path, ["epoch", "train_loss", "train_acc", "test_acc"])

    best_test_accuracy = 0.0

    for epoch in range(1, number_of_epochs + 1):

        student_model.train()
        total_loss          = 0.0
        correct_predictions = 0
        total_samples       = 0

        progress_bar = tqdm(training_loader, desc = f"Scratch Epoch {epoch}/{number_of_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = student_model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss          = total_loss + loss.item()
            predicted_classes   = logits.argmax(dim = 1)
            correct_predictions = correct_predictions + (predicted_classes == labels).sum().item()
            total_samples       = total_samples + labels.size(0)

        scheduler.step()

        train_accuracy = 100.0 * correct_predictions / total_samples
        average_loss   = total_loss / len(training_loader)
        test_accuracy  = compute_accuracy(student_model, test_loader, device)

        print(f"Epoch {epoch}: Loss = {average_loss:.4f}  Train Acc = {train_accuracy:.2f}%  Test Acc = {test_accuracy:.2f}%")

        append_csv_row(log_file_path, [epoch, average_loss, train_accuracy, test_accuracy])

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, "student_scratch.pth"))

    print(f"Best student (no teacher) test accuracy: {best_test_accuracy:.2f}%")

train_student_scratch()

# %%
# Cell 7 — Train student with KD (Hinton)

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def train_student_kd(temperature = 4.0, alpha = 0.9, teacher_checkpoint = "teacher_resnet110.pth", run_name = "student_kd"):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, teacher_checkpoint)
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
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

            with torch.no_grad():
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, f"{run_name}.pth"))

    print(f"Best student (KD) test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

train_student_kd(temperature = 4.0, alpha = 0.9)

# %%
# Cell 8 — Train student with FitNets

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss, hint_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

class HintRegressor(nn.Module):

    def __init__(self, student_channels, teacher_channels):
        super(HintRegressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.regressor(x)

def train_stage1_hints(student_model, teacher_model, training_loader, device, number_of_stage1_epochs, results_directory):

    print("Stage 1: Hint-based pre-training")

    student_hint_channels = 32
    teacher_hint_channels = 32

    regressor = HintRegressor(student_hint_channels, teacher_hint_channels).to(device)

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

            with torch.no_grad():
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

            with torch.no_grad():
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, "student_fitnet.pth"))

    print(f"Best FitNet student test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

def train_student_fitnet(temperature = 4.0, alpha = 0.9):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, "teacher_resnet110.pth")
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    student_model = build_resnet20().to(device)

    train_stage1_hints(
        student_model, teacher_model, training_loader,
        device,
        number_of_stage1_epochs = 50,
        results_directory       = results_directory
    )

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

train_student_fitnet(temperature = 4.0, alpha = 0.9)

# %%
# Cell 9 - Actual

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def train_student_kd(temperature = 4.0, alpha = 0.9, teacher_checkpoint = "teacher_resnet110.pth", run_name = "student_kd"):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, teacher_checkpoint)
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 80], gamma = 0.1)

    number_of_epochs = 100
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

            with torch.no_grad():
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, f"{run_name}.pth"))

    print(f"Best student (KD) test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

temperature_values = [2, 8]
alpha_values       = [0.1, 0.9]

log_file_path = "./results/ablation_temperature.csv"
initialize_csv_log(log_file_path, ["temperature", "alpha", "best_test_acc"])

for temperature in temperature_values:
    for alpha in alpha_values:
        run_name      = f"kd_T{temperature}_a{int(alpha * 10)}"
        print(f"\nRunning: Temperature = {temperature}, Alpha = {alpha}")
        best_accuracy = train_student_kd(
            temperature        = temperature,
            alpha              = alpha,
            teacher_checkpoint = "teacher_resnet110.pth",
            run_name           = run_name
        )
        append_csv_row(log_file_path, [temperature, alpha, best_accuracy])
        print(f"T = {temperature}, alpha = {alpha}: Best Accuracy = {best_accuracy:.2f}%")

# %%
# Cell 10 - Actual

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def train_student_kd(temperature = 4.0, alpha = 0.9, teacher_checkpoint = "teacher_resnet110.pth", run_name = "student_kd"):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, teacher_checkpoint)
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 80], gamma = 0.1)

    number_of_epochs = 100
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

            with torch.no_grad():
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, f"{run_name}.pth"))

    print(f"Best student (KD) test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

checkpoint_directory      = "./checkpoints"
teacher_checkpoint_epochs = [50, 100, 150, 200]

log_file_path = "./results/ablation_early_stop.csv"
initialize_csv_log(log_file_path, ["teacher_epoch", "best_test_acc"])

for teacher_epoch in teacher_checkpoint_epochs:

    if teacher_epoch == 200:
        checkpoint_file = "teacher_resnet110.pth"
    else:
        checkpoint_file = f"teacher_epoch_{teacher_epoch}.pth"

    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}, skipping")
        continue

    run_name = f"kd_teacher_epoch_{teacher_epoch}"
    print(f"\nUsing teacher checkpoint at epoch {teacher_epoch}")

    best_accuracy = train_student_kd(
        temperature        = 4.0,
        alpha              = 0.9,
        teacher_checkpoint = checkpoint_file,
        run_name           = run_name
    )

    append_csv_row(log_file_path, [teacher_epoch, best_accuracy])
    print(f"Teacher epoch {teacher_epoch}: Best Student Accuracy = {best_accuracy:.2f}%")

# %%
# Cell 10 - 200 Only (in case of GPU limit)

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

from src.dataset import get_cifar100_loaders
from src.models import build_resnet110, build_resnet20
from src.losses import knowledge_distillation_loss
from src.utils import compute_accuracy, save_checkpoint, load_checkpoint, initialize_csv_log, append_csv_row, get_device

import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def train_student_kd(temperature = 4.0, alpha = 0.9, teacher_checkpoint = "teacher_resnet110.pth", run_name = "student_kd"):

    device = get_device()

    data_directory       = "./data"
    checkpoint_directory = "./checkpoints"
    results_directory    = "./results"

    os.makedirs(checkpoint_directory, exist_ok = True)
    os.makedirs(results_directory,    exist_ok = True)

    training_loader, test_loader = get_cifar100_loaders(data_directory, batch_size = 128)

    teacher_model           = build_resnet110().to(device)
    teacher_checkpoint_path = os.path.join(checkpoint_directory, teacher_checkpoint)
    teacher_model           = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 80], gamma = 0.1)

    number_of_epochs = 100
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

            with torch.no_grad():
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            save_checkpoint(student_model, os.path.join(checkpoint_directory, f"{run_name}.pth"))

    print(f"Best student (KD) test accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy

checkpoint_directory      = "./checkpoints"
teacher_checkpoint_epochs = [200]

log_file_path = "./results/ablation_early_stop.csv"
initialize_csv_log(log_file_path, ["teacher_epoch", "best_test_acc"])

for teacher_epoch in teacher_checkpoint_epochs:

    if teacher_epoch == 200:
        checkpoint_file = "teacher_resnet110.pth"
    else:
        checkpoint_file = f"teacher_epoch_{teacher_epoch}.pth"

    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}, skipping")
        continue

    run_name = f"kd_teacher_epoch_{teacher_epoch}"
    print(f"\nUsing teacher checkpoint at epoch {teacher_epoch}")

    best_accuracy = train_student_kd(
        temperature        = 4.0,
        alpha              = 0.9,
        teacher_checkpoint = checkpoint_file,
        run_name           = run_name
    )

    append_csv_row(log_file_path, [teacher_epoch, best_accuracy])
    print(f"Teacher epoch {teacher_epoch}: Best Student Accuracy = {best_accuracy:.2f}%")

# %%
# Cell 11 — Visualizations (all plots render inline and save to Drive)

import os, sys
os.chdir("/content/drive/MyDrive/25280058_Project")
sys.path.insert(0, "/content/drive/MyDrive/25280058_Project")

import pandas as pd
import matplotlib.pyplot as plt

figures_directory = "./figures"
results_directory = "./results"
os.makedirs(figures_directory, exist_ok = True)

# Plot 1: Test accuracy curves for all four training runs

figure, axis = plt.subplots(figsize = (10, 6))

log_files = {
    "Teacher (ResNet-110)":  "teacher_log.csv",
    "Student - No Teacher":  "student_scratch_log.csv",
    "Student - KD (Hinton)": "student_kd_log.csv",
    "Student - FitNet":      "student_fitnet_log.csv"
}

for label, filename in log_files.items():
    file_path = os.path.join(results_directory, filename)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        axis.plot(data["epoch"], data["test_acc"], label = label)

axis.set_xlabel("Epoch")
axis.set_ylabel("Test Accuracy (%)")
axis.set_title("Test Accuracy over Training on CIFAR-100")
axis.legend()
axis.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_directory, "accuracy_curves.png"), dpi = 150)
plt.show()
print("Saved: accuracy_curves.png")

# Plot 2: Temperature ablation

file_path = os.path.join(results_directory, "ablation_temperature.csv")
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    figure, axis = plt.subplots(figsize = (9, 5))
    for alpha_value in sorted(data["alpha"].unique()):
        subset = data[data["alpha"] == alpha_value]
        axis.plot(subset["temperature"], subset["best_test_acc"],
                  marker = "o", label = f"alpha = {alpha_value}")
    axis.set_xlabel("Temperature (T)")
    axis.set_ylabel("Best Test Accuracy (%)")
    axis.set_title("Effect of Temperature and Alpha on Student Accuracy")
    axis.legend()
    axis.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory, "ablation_temperature.png"), dpi = 150)
    plt.show()
    print("Saved: ablation_temperature.png")

# Plot 3: Early stopping ablation

file_path = os.path.join(results_directory, "ablation_early_stop.csv")
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    figure, axis = plt.subplots(figsize = (8, 5))
    axis.plot(data["teacher_epoch"], data["best_test_acc"], marker = "s", color = "steelblue")
    axis.set_xlabel("Teacher Training Epoch (checkpoint used)")
    axis.set_ylabel("Best Student Test Accuracy (%)")
    axis.set_title("Effect of Teacher Early Stopping on Student Accuracy")
    axis.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory, "ablation_early_stop.png"), dpi = 150)
    plt.show()
    print("Saved: ablation_early_stop.png")

# Summary table

print("\nResults Summary")
print("-" * 50)
log_files = {
    "Teacher (ResNet-110)":  "teacher_log.csv",
    "Student - No Teacher":  "student_scratch_log.csv",
    "Student - KD (Hinton)": "student_kd_log.csv",
    "Student - FitNet":      "student_fitnet_log.csv"
}
for label, filename in log_files.items():
    file_path = os.path.join(results_directory, filename)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        best_accuracy = data["test_acc"].max()
        print(f"{label}: {best_accuracy:.2f}%")
print("-" * 50)


