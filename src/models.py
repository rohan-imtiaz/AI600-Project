# AI 600 Project 15: Knowledge Distillation (models.py)

# INSTRUCTIONS:
# This file is not run directly.
# It is imported by all training scripts to provide ResNet-110 and ResNet-20.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic residual block used in all CIFAR ResNet variants

class ResidualBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            input_channels, output_channels,
            kernel_size = 3, stride = stride, padding = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(
            output_channels, output_channels,
            kernel_size = 3, stride = 1, padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

        # Shortcut connection handles dimension changes between stages
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels,
                    kernel_size = 1, stride = stride, bias = False
                ),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# General CIFAR ResNet class (depth 20 for student, depth 110 for teacher)

class ResNet(nn.Module):

    def __init__(self, depth, num_classes = 100):
        super(ResNet, self).__init__()

        assert (depth - 2) % 6 == 0, "depth must satisfy (depth - 2) mod 6 == 0"
        number_of_blocks = (depth - 2) // 6

        self.current_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1   = nn.BatchNorm2d(16)

        # Three stages with channel widths 16, 32, 64
        self.stage1 = self._make_stage(16, number_of_blocks, stride = 1)
        self.stage2 = self._make_stage(32, number_of_blocks, stride = 2)
        self.stage3 = self._make_stage(64, number_of_blocks, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64, num_classes)

        # Kaiming initialization for conv layers, constant init for batch norm
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_stage(self, output_channels, number_of_blocks, stride):
        strides = [stride] + [1] * (number_of_blocks - 1)
        layers  = []
        for stride_value in strides:
            layers.append(ResidualBlock(self.current_channels, output_channels, stride_value))
            self.current_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        hint_features = out # Saving stage2 output as intermediate hint features for FitNets
        out = self.stage3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits, hint_features

    def get_hint_features(self, x): # Returns stage2 output for hint-based training only
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        return out

# Convenience constructors for teacher and student

def build_resnet110():
    return ResNet(depth = 110, num_classes = 100)

def build_resnet20():
    return ResNet(depth = 20, num_classes = 100)
