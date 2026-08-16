#!/usr/bin/env python3
"""Three-class, mask-assisted CTMF-Net adaptation for the common BUSI protocol."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ConvNormAct, FeedForward, run_experiment


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = torch.sigmoid(
            self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
            + self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        )
        x = x * channel
        spatial = torch.sigmoid(
            self.spatial(torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1))
        )
        return x * spatial


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class FeatureInteractionModule(nn.Module):
    """Transformer queries guide CNN key/value features, repeated twice as in the paper."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.cnn_norms = nn.ModuleList([nn.LayerNorm(channels), nn.LayerNorm(channels)])
        self.transformer_norms = nn.ModuleList([nn.LayerNorm(channels), nn.LayerNorm(channels)])
        self.cross_attention = nn.ModuleList(
            [nn.MultiheadAttention(channels, heads, dropout=0.1, batch_first=True) for _ in range(2)]
        )
        self.mlps = nn.ModuleList([FeedForward(channels, expansion=2) for _ in range(2)])
        self.output = ConvNormAct(channels, channels)

    def forward(self, cnn: torch.Tensor, transformer: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cnn.shape
        local = cnn.flatten(2).transpose(1, 2)
        global_tokens = transformer.flatten(2).transpose(1, 2)
        for cnn_norm, transformer_norm, attention, mlp in zip(
            self.cnn_norms, self.transformer_norms, self.cross_attention, self.mlps
        ):
            query = transformer_norm(global_tokens)
            key_value = cnn_norm(local)
            local = local + attention(query, key_value, key_value, need_weights=False)[0]
            local = local + mlp(cnn_norm(local))
        interacted = local.transpose(1, 2).reshape(batch, channels, height, width)
        return self.output(F.gelu(cnn + interacted))


class CTMFNet(nn.Module):
    """Compact four-stage VGG/ViT fusion preserving CTMF-Net's defining modules."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        channels = (32, 64, 96, 128)
        heads = (2, 4, 4, 4)
        self.cnn_stem = nn.Sequential(
            ConvNormAct(3, 24, stride=2),
            ConvNormAct(24, channels[0], stride=2),
            ConvNormAct(channels[0], channels[0], stride=2),
        )
        self.transformer_stem = nn.Conv2d(3, channels[0], kernel_size=8, stride=8)
        self.cnn_stages = nn.ModuleList()
        self.transformer_stages = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.cnn_downsamples = nn.ModuleList()
        self.transformer_downsamples = nn.ModuleList()
        for index, (dim, num_heads) in enumerate(zip(channels, heads)):
            self.cnn_stages.append(nn.Sequential(ConvNormAct(dim, dim), ConvNormAct(dim, dim)))
            self.transformer_stages.append(
                nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(3)])
            )
            self.cbams.append(CBAM(dim))
            self.interactions.append(FeatureInteractionModule(dim, num_heads))
            if index < len(channels) - 1:
                self.cnn_downsamples.append(ConvNormAct(dim, channels[index + 1], stride=2))
                self.transformer_downsamples.append(
                    nn.Conv2d(dim, channels[index + 1], kernel_size=2, stride=2)
                )
        self.cnn_norm = nn.LayerNorm(channels[-1])
        self.transformer_norm = nn.LayerNorm(channels[-1])
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 2, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn_stem(x)
        transformer = self.transformer_stem(x)
        for index in range(4):
            cnn = self.cbams[index](self.cnn_stages[index](cnn))
            batch, channels, height, width = transformer.shape
            tokens = transformer.flatten(2).transpose(1, 2)
            for block in self.transformer_stages[index]:
                tokens = block(tokens)
            transformer = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            cnn = self.interactions[index](cnn, transformer)
            if index < 3:
                cnn = self.cnn_downsamples[index](cnn)
                transformer = self.transformer_downsamples[index](transformer)
        cnn_vector = self.cnn_norm(F.adaptive_avg_pool2d(cnn, 1).flatten(1))
        transformer_vector = self.transformer_norm(
            F.adaptive_avg_pool2d(transformer, 1).flatten(1)
        )
        return self.classifier(torch.cat((cnn_vector, transformer_vector), dim=1))


if __name__ == "__main__":
    run_experiment(
        model_key="ctmf_net",
        display_name="CTMF-Net",
        implementation_scope=(
            "Three-class 128x128 adaptation of Wang et al.: four VGG-like CNN stages with CBAM, "
            "four three-block ViT stages, and two-pass cross-attention FIMs. The output head and "
            "input channels are adapted to the shared BUSI protocol."
        ),
        build_model=lambda num_classes: CTMFNet(num_classes),
    )

