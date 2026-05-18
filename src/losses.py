# AI 600 Project 15: Knowledge Distillation (losses.py)

# INSTRUCTIONS:
# This file is not run directly.
# It is imported by all student training scripts to provide the KD and hint loss functions.

import torch.nn.functional as F

# Hinton et al. combined hard and soft knowledge distillation loss

def knowledge_distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):

    # Soft loss: KL divergence between teacher and student softened distributions
    # Scaling by temperature squared keeps gradient magnitudes consistent across temperatures (Hinton et al.)
    soft_teacher_distribution = F.softmax(teacher_logits / temperature, dim = 1)
    soft_student_log           = F.log_softmax(student_logits / temperature, dim = 1)
    soft_loss                  = F.kl_div(soft_student_log, soft_teacher_distribution, reduction = "batchmean")
    soft_loss                  = soft_loss * (temperature * temperature)

    # Hard loss: standard cross entropy against ground truth labels
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # Combining losses: alpha weights the hard label term, (1 - alpha) weights the soft term
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss

# FitNets hint loss: L2 distance between projected student features and teacher hint features

def hint_loss(student_hint_features, teacher_hint_features, regressor):

    projected_student_features = regressor(student_hint_features) # Projecting student features to teacher channel size
    loss = F.mse_loss(projected_student_features, teacher_hint_features.detach())
    return loss