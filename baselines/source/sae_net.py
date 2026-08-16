#!/usr/bin/env python3
"""Image-spectrum SAE-Net adaptation for BUSI, where raw RF is unavailable."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ConvNormAct, decoded_channels, run_experiment


class DenseUnit(nn.Module):
    def __init__(self, in_channels: int, growth: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, growth * 2, 1, bias=False),
            nn.BatchNorm2d(growth * 2),
            nn.GELU(),
            nn.Conv2d(growth * 2, growth, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.block(x)), dim=1)


class ImageEncoder(nn.Module):
    """Small DenseNet-like grayscale morphology encoder."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(3, 32, stride=2), nn.MaxPool2d(2))
        channels = 32
        stages = []
        for units, growth in ((3, 16), (4, 20), (4, 24)):
            dense = []
            for _ in range(units):
                dense.append(DenseUnit(channels, growth))
                channels += growth
            output_channels = min(channels, 160)
            stages.extend(
                [
                    nn.Sequential(*dense),
                    nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.GELU(),
                        nn.Conv2d(channels, output_channels, 1, bias=False),
                        nn.AvgPool2d(2),
                    ),
                ]
            )
            channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.projection = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 192))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.projection(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ImageSpectrumEncoder(nn.Module):
    """Spectral-statistics substitute for the unavailable RF CWT branch.

    The paper's cumulative distribution of tissue malignancy probability is
    retained. Each overlapping lesion-texture patch is described by radial FFT
    band means/standard deviations, classified by an MLP, and accumulated at
    128 thresholds. This is intentionally labelled as an adaptation, since a
    B-mode PNG cannot recover the paper's 50 MHz RF waveform.
    """

    def __init__(self, patch_size: int = 16, stride: int = 8, bands: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.bands = bands
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch_size),
            torch.linspace(0.0, 1.0, patch_size // 2 + 1),
            indexing="ij",
        )
        radius = torch.sqrt(xx.square() + yy.square())
        edges = torch.linspace(0.0, float(radius.max()) + 1e-6, bands + 1)
        masks = []
        for index in range(bands):
            mask = ((radius >= edges[index]) & (radius < edges[index + 1])).float()
            masks.append(mask / mask.sum().clamp_min(1.0))
        self.register_buffer("radial_masks", torch.stack(masks))
        self.patch_classifier = nn.Sequential(
            nn.Linear(bands * 2 + 2, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )
        self.register_buffer("thresholds", torch.linspace(0.0, 1.0, 128))

    def forward(self, normalized_input: torch.Tensor) -> torch.Tensor:
        decoded = decoded_channels(normalized_input)
        lesion_texture = decoded[:, 2:3]
        lesion_mask = decoded[:, 1:2]
        patches = F.unfold(
            lesion_texture, kernel_size=self.patch_size, stride=self.stride
        ).transpose(1, 2)
        mask_patches = F.unfold(
            lesion_mask, kernel_size=self.patch_size, stride=self.stride
        ).transpose(1, 2)
        batch, patch_count, _ = patches.shape
        patch_images = patches.reshape(batch * patch_count, self.patch_size, self.patch_size)
        spectrum = torch.log1p(torch.abs(torch.fft.rfft2(patch_images, norm="ortho")))
        # B,P,K: mean energy in each fixed radial band.
        band_mean = torch.einsum("bxy,kxy->bk", spectrum, self.radial_masks)
        centered = spectrum[:, None] - band_mean[:, :, None, None]
        band_var = torch.einsum("bkxy,kxy->bk", centered.square(), self.radial_masks)
        mask_fraction = mask_patches.mean(dim=-1).reshape(-1, 1)
        intensity_mean = patches.mean(dim=-1).reshape(-1, 1)
        statistics = torch.cat(
            (band_mean, torch.sqrt(band_var + 1e-6), mask_fraction, intensity_mean), dim=1
        )
        tissue_logits = self.patch_classifier(statistics).reshape(batch, patch_count, 3)
        malignant_probability = tissue_logits.softmax(dim=-1)[..., 1]
        valid = (mask_fraction.reshape(batch, patch_count) > 0.01).float()
        # Smooth unit-step approximation to the paper's CDCM.
        cumulative = torch.sigmoid(
            24.0 * (malignant_probability[..., None] - self.thresholds)
        )
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        cumulative = (cumulative * valid[..., None]).sum(dim=1) / denominator
        # Normal images have no valid lesion patch and therefore retain a zero CDCM.
        return cumulative


class SAENet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.spectrum_encoder = ImageSpectrumEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(192 + 128, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        morphology = self.image_encoder(x)
        spectrum = self.spectrum_encoder(x)
        return self.classifier(torch.cat((morphology, spectrum), dim=1))


if __name__ == "__main__":
    run_experiment(
        model_key="sae_net",
        display_name="SAE-Net (image-spectrum adaptation)",
        implementation_scope=(
            "Non-equivalent image-only adaptation of Xie et al. The grayscale DenseNet-like branch and "
            "128-bin cumulative tissue-probability pooling are retained, but radial B-mode FFT statistics "
            "replace the unavailable raw-RF bump-wavelet spectra. Results must not be described as an exact "
            "SAE-Net reproduction."
        ),
        build_model=lambda num_classes: SAENet(num_classes),
    )

