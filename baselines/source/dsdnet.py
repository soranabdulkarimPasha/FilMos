#!/usr/bin/env python3
"""Mask-prior DSDNet adaptation for the common three-class BUSI protocol."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ConvNormAct, FeedForward, decoded_channels, run_experiment


class DilatedContext(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch_channels = max(channels // 3, 4)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, branch_channels, 3, padding=dilation, dilation=dilation)
                for dilation in (3, 5, 7)
            ]
        )
        self.output = ConvNormAct(branch_channels * 3, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.cat([branch(x) for branch in self.branches], dim=1))


class DSMA(nn.Module):
    """Dual foreground/background mask attention with 3/5/7 dilated context."""

    def __init__(self, channels: int):
        super().__init__()
        self.prior_projection = nn.Sequential(
            nn.Conv2d(1, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.GELU()
        )
        self.foreground_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.foreground_context = DilatedContext(channels)
        self.background_context = DilatedContext(channels)
        self.output = ConvNormAct(channels * 2, channels, kernel_size=1)

    def forward(self, mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
        prior = self.prior_projection(mask)
        foreground_map = self.foreground_attention(prior)
        background_map = 1.0 - foreground_map
        foreground = self.foreground_context(prior * foreground_map)
        background = self.background_context(prior * background_map)
        return self.output(torch.cat((foreground, background), dim=1))


class SegmentationPriorGuidanceFusion(nn.Module):
    """Segmentation-modulated token attention and residual classifier fusion."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.prior_projection = nn.Conv2d(channels, channels, 1)
        self.depthwise = ConvNormAct(channels * 2, channels, groups=1)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.1, batch_first=True)
        self.output = ConvNormAct(channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = self.prior_projection(prior)
        gate = torch.sigmoid(prior)
        fused = self.depthwise(torch.cat((features, features * gate), dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        prior_tokens = gate.flatten(2).transpose(1, 2)
        query = self.norm(tokens * (1.0 + prior_tokens))
        attended = self.attention(query, query, tokens, need_weights=False)[0]
        attended = attended.transpose(1, 2).reshape_as(features)
        return features + self.output(attended)


class LinearAttention(nn.Module):
    """ELU+1 linear attention with depth-wise local positional encoding."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.qk = nn.Linear(channels, channels * 2)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.lepe = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k = self.qk(x).chunk(2, dim=-1)
        v = self.value(x)
        q = (F.elu(q) + 1.0).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = (F.elu(k) + 1.0).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v_heads = v.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        kv = torch.einsum("bhnd,bhne->bhde", k, v_heads)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
        attended = torch.einsum("bhnd,bhde->bhne", q, kv) / normalizer[..., None]
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        local = self.lepe(v.transpose(1, 2).reshape(batch, channels, height, width))
        local = local.flatten(2).transpose(1, 2)
        return self.output(attended + local)


class MILTBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = LinearAttention(channels, heads)
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.SiLU())
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForward(channels, expansion=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended = self.attention(normalized, height, width) * self.gate(normalized)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class DSDNet(nn.Module):
    """Four-level DSMA -> SPGF -> MILT classifier using the shared ground-truth mask."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        channels = (32, 64, 96, 128)
        heads = (2, 4, 4, 4)
        self.stem = nn.Sequential(
            ConvNormAct(3, 24, stride=2),
            ConvNormAct(24, channels[0], stride=2),
            ConvNormAct(channels[0], channels[0], stride=2),
        )
        self.stages = nn.ModuleList()
        self.dsma = nn.ModuleList()
        self.spgf = nn.ModuleList()
        self.milt = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (dim, num_heads) in enumerate(zip(channels, heads)):
            self.stages.append(nn.Sequential(ConvNormAct(dim, dim), ConvNormAct(dim, dim)))
            self.dsma.append(DSMA(dim))
            self.spgf.append(SegmentationPriorGuidanceFusion(dim, num_heads))
            self.milt.append(MILTBlock(dim, num_heads))
            if index < 3:
                self.downsamples.append(ConvNormAct(dim, channels[index + 1], stride=2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = decoded_channels(x)[:, 1:2]
        features = self.stem(x)
        for index in range(4):
            features = self.stages[index](features)
            prior = self.dsma[index](mask, features.shape[-2:])
            features = self.spgf[index](features, prior)
            features = self.milt[index](features)
            if index < 3:
                features = self.downsamples[index](features)
        vector = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(vector)


if __name__ == "__main__":
    run_experiment(
        model_key="dsdnet",
        display_name="DSDNet (same-mask adaptation)",
        implementation_scope=(
            "Three-class adaptation preserving four DSMA, SPGF and MILT stages. For information "
            "parity with mask-assisted FiLMoS-Net, the supplied BUSI union mask is the frozen spatial "
            "prior; this replaces the paper's externally pretrained BUSBRA segmentation network. The "
            "four guided scales begin at 16x16 for the shared 128x128 input. This must be reported as "
            "a same-protocol adaptation."
        ),
        build_model=lambda num_classes: DSDNet(num_classes),
    )
