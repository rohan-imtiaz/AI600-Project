# AI 600 Project 15: Knowledge Distillation (utils.py)

# INSTRUCTIONS:
# This file is not run directly.
# It is imported by all scripts for accuracy computation, checkpointing, CSV logging, and device setup.

import os
import csv
import torch

# Computing top-1 accuracy of a model over a given data loader

def compute_accuracy(model, data_loader, device):

    model.eval()
    correct_predictions = 0
    total_samples       = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            predicted_classes   = logits.argmax(dim = 1)
            correct_predictions = correct_predictions + (predicted_classes == labels).sum().item()
            total_samples       = total_samples + labels.size(0)

    accuracy = 100.0 * correct_predictions / total_samples
    return accuracy

# Saving model state dict to disk

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

# Loading model state dict from disk

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location = device))
    print(f"Checkpoint loaded from {path}")
    return model

# Creating a new CSV log file with a header row

def initialize_csv_log(file_path, column_names):
    with open(file_path, "w", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)

# Appending a single row of values to an existing CSV log

def append_csv_row(file_path, row_values):
    with open(file_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row_values)

# Detecting and returning the best available compute device

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device