# AI 600 Project 15: Knowledge Distillation (ablation_temperature.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/ablation_temperature.py
# Sweeps temperature T in {1, 2, 4, 8, 16, 20} and alpha in {0.1, 0.5, 0.9}.
# Each combination trains a fresh student and logs its best accuracy to results/ablation_temperature.csv.
# Teacher checkpoint must already exist at checkpoints/teacher_resnet110.pth.
# Note: this script is computationally heavy. Run with T in {2, 4, 8} and alpha = 0.9 if time is limited.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.train_student_kd import train_student_kd
from src.utils import initialize_csv_log, append_csv_row

def run_temperature_ablation():

    results_directory = "./results"
    os.makedirs(results_directory, exist_ok = True)

    log_file_path = os.path.join(results_directory, "ablation_temperature.csv")
    initialize_csv_log(log_file_path, ["temperature", "alpha", "best_test_acc"])

    # Temperature values follow the range used in Hinton et al. and Cho & Hariharan
    temperature_values = [1, 2, 4, 8, 16, 20]

    # Alpha controls the weight on the hard label cross entropy term
    alpha_values = [0.1, 0.5, 0.9]

    for temperature in temperature_values:
        for alpha in alpha_values:

            run_name = f"kd_T{temperature}_a{int(alpha * 10)}"
            print(f"\nRunning: Temperature = {temperature}, Alpha = {alpha}")

            best_accuracy = train_student_kd(
                temperature        = temperature,
                alpha              = alpha,
                teacher_checkpoint = "teacher_resnet110.pth",
                run_name           = run_name
            )

            append_csv_row(log_file_path, [temperature, alpha, best_accuracy])
            print(f"T = {temperature}, alpha = {alpha}: Best Accuracy = {best_accuracy:.2f}%")

if __name__ == "__main__":
    run_temperature_ablation()