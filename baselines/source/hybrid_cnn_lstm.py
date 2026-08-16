#!/usr/bin/env python3
"""Hybrid CNN-BiLSTM baseline for the common BUSI protocol."""

from __future__ import annotations

import torch
import torch.nn as nn

from common import ConvNormAct, run_experiment


class HybridCNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvNormAct(3, 32, stride=2),
            ConvNormAct(32, 32),
            ConvNormAct(32, 64, stride=2),
            ConvNormAct(64, 64),
            ConvNormAct(64, 96, stride=2),
            ConvNormAct(96, 128, stride=2),
        )
        self.sequence_projection = nn.Linear(128, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=96,
            num_layers=2,
            dropout=0.15,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(192), nn.Dropout(0.2), nn.Linear(192, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        # Each spatial position is a sequence element; the BiLSTM learns both
        # forward and backward dependencies over the CNN feature grid.
        sequence = features.flatten(2).transpose(1, 2)
        sequence = self.sequence_projection(sequence)
        sequence, _ = self.lstm(sequence)
        weights = torch.softmax(self.attention(sequence), dim=1)
        vector = (sequence * weights).sum(dim=1)
        return self.classifier(vector)


if __name__ == "__main__":
    run_experiment(
        model_key="hybrid_cnn_lstm",
        display_name="Hybrid CNN-LSTM",
        implementation_scope=(
            "Four-scale CNN feature extractor followed by a two-layer bidirectional LSTM and "
            "learned attention pooling over the spatial feature sequence; three-class output."
        ),
        build_model=lambda num_classes: HybridCNNLSTM(num_classes),
    )

