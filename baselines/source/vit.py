#!/usr/bin/env python3
"""Standalone Vision Transformer baseline for the common BUSI protocol."""

from __future__ import annotations

import torch
import torch.nn as nn

from common import FeedForward, run_experiment


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        image_size: int = 128,
        patch_size: int = 16,
        dim: int = 128,
        depth: int = 6,
        heads: int = 4,
    ):
        super().__init__()
        grid = image_size // patch_size
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.blocks = nn.Sequential(*[ViTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, num_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1) + self.position
        return self.head(self.norm(self.blocks(tokens))[:, 0])


if __name__ == "__main__":
    run_experiment(
        model_key="vit",
        display_name="ViT",
        implementation_scope=(
            "Standalone ViT-B/16-style baseline with 16x16 patches, six encoder blocks, 128-dimensional "
            "tokens and four attention heads. This implements the requested ViT model, not a "
            "multi-checkpoint ensemble."
        ),
        build_model=lambda num_classes: VisionTransformer(num_classes),
    )
