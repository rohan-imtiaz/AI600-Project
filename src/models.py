# AI 600 Project 15: Knowledge Distillation (models.py)

# INSTRUCTIONS:
# This file is not run directly.
# It is imported by all training scripts to provide ResNet-110, ResNet-20, and MobileNetV2.

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

# Inverted residual block used inside MobileNetV2

class InvertedResidual(nn.Module):

    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        hidden_channels = int(input_channels * expand_ratio)
        self.use_residual = (stride == 1 and input_channels == output_channels)

        layers = []

        if expand_ratio != 1: # Pointwise expansion only when ratio is not 1
            layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size = 1, bias = False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU6(inplace = True))

        layers.append(nn.Conv2d(
            hidden_channels, hidden_channels,
            kernel_size = 3, stride = stride, padding = 1,
            groups = hidden_channels, bias = False
        ))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU6(inplace = True))
        layers.append(nn.Conv2d(hidden_channels, output_channels, kernel_size = 1, bias = False))
        layers.append(nn.BatchNorm2d(output_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV2 adapted for CIFAR-100 (alternative student model)

class MobileNetV2(nn.Module):

    def __init__(self, num_classes = 100):
        super(MobileNetV2, self).__init__()

        # Each tuple is (expand_ratio, output_channels, num_blocks, stride)
        inverted_residual_settings = [
            (1, 16,  1, 1),
            (6, 24,  2, 1),
            (6, 32,  3, 2),
            (6, 64,  4, 2),
            (6, 96,  3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        input_channel = 32

        feature_layers = [
            nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace = True)
            )
        ]

        for expand_ratio, output_channel, num_blocks, stride in inverted_residual_settings:
            for block_index in range(num_blocks):
                current_stride = stride if block_index == 0 else 1
                feature_layers.append(
                    InvertedResidual(
                        input_channel, output_channel,
                        stride = current_stride,
                        expand_ratio = expand_ratio
                    )
                )
                input_channel = output_channel

        feature_layers.append(nn.Sequential(
            nn.Conv2d(input_channel, 1280, kernel_size = 1, bias = False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace = True)
        ))

        self.features   = nn.Sequential(*feature_layers)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits       = self.classifier(out)
        hint_features = out # Returning flat features as dummy hint for interface compatibility
        return logits, hint_features