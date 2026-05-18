# AI 600 Project 15: Knowledge Distillation (visualize.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/visualize.py
# Reads CSV logs from results/ and saves all figures to figures/.
# Also prints a summary table of best accuracies for all training runs.
# Run this after all training and ablation scripts have completed.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt

# Plotting test accuracy curves for all four training configurations

def plot_accuracy_curves():

    figures_directory = "./figures"
    results_directory = "./results"

    os.makedirs(figures_directory, exist_ok = True)

    figure, axis = plt.subplots(figsize = (10, 6))

    # Each entry maps a display label to its CSV log filename
    log_files = {
        "Teacher (ResNet-110)":    "teacher_log.csv",
        "Student - No Teacher":    "student_scratch_log.csv",
        "Student - KD (Hinton)":   "student_kd_log.csv",
        "Student - FitNet":        "student_fitnet_log.csv"
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

    output_path = os.path.join(figures_directory, "accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150)
    plt.close()
    print(f"Saved accuracy curves to {output_path}")

# Plotting the effect of temperature and alpha on best student accuracy

def plot_temperature_ablation():

    figures_directory = "./figures"
    results_directory = "./results"

    os.makedirs(figures_directory, exist_ok = True)

    file_path = os.path.join(results_directory, "ablation_temperature.csv")
    if not os.path.exists(file_path):
        print("Temperature ablation results not found, skipping plot")
        return

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

    output_path = os.path.join(figures_directory, "ablation_temperature.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150)
    plt.close()
    print(f"Saved temperature ablation plot to {output_path}")

# Plotting the effect of teacher checkpoint age on best student accuracy

def plot_early_stop_ablation():

    figures_directory = "./figures"
    results_directory = "./results"

    os.makedirs(figures_directory, exist_ok = True)

    file_path = os.path.join(results_directory, "ablation_early_stop.csv")
    if not os.path.exists(file_path):
        print("Early stop ablation results not found, skipping plot")
        return

    data = pd.read_csv(file_path)

    figure, axis = plt.subplots(figsize = (8, 5))

    axis.plot(data["teacher_epoch"], data["best_test_acc"], marker = "s", color = "steelblue")
    axis.set_xlabel("Teacher Training Epoch (checkpoint used)")
    axis.set_ylabel("Best Student Test Accuracy (%)")
    axis.set_title("Effect of Teacher Early Stopping on Student Accuracy")
    axis.grid(True, alpha = 0.3)

    output_path = os.path.join(figures_directory, "ablation_early_stop.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150)
    plt.close()
    print(f"Saved early stop ablation plot to {output_path}")

# Printing a summary table of best test accuracies for all main experiments

def print_results_summary():

    results_directory = "./results"
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

if __name__ == "__main__":
    plot_accuracy_curves()
    plot_temperature_ablation()
    plot_early_stop_ablation()
    print_results_summary()