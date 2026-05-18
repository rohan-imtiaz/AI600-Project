# AI 600 Project 15: Knowledge Distillation (ablation_early_stop.py)

# INSTRUCTIONS:
# Run from the 25280058_Project directory:
# python src/ablation_early_stop.py
# Tests whether using an early-stopped teacher checkpoint produces a better student (Cho & Hariharan 2019).
# For each saved teacher snapshot, a fresh student is trained with KD and its best accuracy is logged.
# All intermediate teacher checkpoints must already exist (saved during train_teacher.py).
# Results are written to results/ablation_early_stop.csv.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.train_student_kd import train_student_kd
from src.utils import initialize_csv_log, append_csv_row

def run_early_stop_ablation():

    results_directory    = "./results"
    checkpoint_directory = "./checkpoints"

    os.makedirs(results_directory, exist_ok = True)

    log_file_path = os.path.join(results_directory, "ablation_early_stop.csv")
    initialize_csv_log(log_file_path, ["teacher_epoch", "best_test_acc"])

    # Intermediate checkpoints saved during train_teacher.py plus the final best checkpoint
    teacher_checkpoint_epochs = [50, 80, 100, 130, 150, 200]

    for teacher_epoch in teacher_checkpoint_epochs:

        if teacher_epoch == 200: # The final best checkpoint uses a different filename
            checkpoint_file = "teacher_resnet110.pth"
        else:
            checkpoint_file = f"teacher_epoch_{teacher_epoch}.pth"

        checkpoint_path = os.path.join(checkpoint_directory, checkpoint_file)

        if not os.path.exists(checkpoint_path): # Skipping if this checkpoint was not saved
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

if __name__ == "__main__":
    run_early_stop_ablation()
