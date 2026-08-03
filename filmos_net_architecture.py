"""FiLMoS-Net architecture used in the revised manuscript.

The module exposes the learnable Gabor, fixed DCT, differentiable morphology,
adaptive routing, fusion, and classification components as ordinary PyTorch
classes. It is also preserved verbatim in the complete experiment notebook.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 2) FiLMoS-Net: Filtered Multi-scale Morphology + Spectral Net
#    Branches: (1) Learnable Gabor bank  (2) Fixed DCT spectral
#              (3) Edge + Soft Morphology
#    Fusion: Router produces per-sample weights for branches
# ============================================================

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] -> gray: [B,1,H,W]
    """
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------
# Branch 1: Parametric Gabor Bank (learnable parameters -> generated kernels)
# ---------------------------
class LearnableGaborBank(nn.Module):
    """
    Generates N gabor kernels (single-channel) with learnable parameters.
    Applies on grayscale input: [B,1,H,W] -> [B,N,H,W]
    """
    def __init__(self, n_filters: int = 32, ksize: int = 21):
        super().__init__()
        if ksize % 2 == 0:
            raise ValueError("Gabor ksize must be odd.")
        self.n_filters = n_filters
        self.ksize = ksize

        # raw parameters (unconstrained) -> mapped to valid ranges in forward
        self.theta_raw = nn.Parameter(torch.randn(n_filters) * 0.3)   # orientation
        self.freq_raw  = nn.Parameter(torch.randn(n_filters) * 0.3)   # spatial frequency
        self.sigma_raw = nn.Parameter(torch.randn(n_filters) * 0.3)   # gaussian std
        self.gamma_raw = nn.Parameter(torch.randn(n_filters) * 0.3)   # aspect ratio
        self.psi_raw   = nn.Parameter(torch.randn(n_filters) * 0.3)   # phase
        self.amp_raw   = nn.Parameter(torch.zeros(n_filters))         # amplitude scaling

        # precompute coordinate grid (registered buffer)
        half = ksize // 2
        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).float()  # [2,ks,ks]
        self.register_buffer("grid", grid, persistent=False)

    def _build_kernels(self, device, dtype) -> torch.Tensor:
        """
        Returns kernels: [N,1,ks,ks]
        """
        grid = self.grid.to(device=device, dtype=dtype)
        x = grid[0]  # [ks,ks]
        y = grid[1]

        # map params to ranges
        # theta in [0, pi)
        theta = torch.sigmoid(self.theta_raw) * math.pi

        # freq in [0.05, 0.45] cycles/pixel (reasonable for 224 images)
        freq = 0.05 + torch.sigmoid(self.freq_raw) * (0.45 - 0.05)

        # sigma in [2.0, 8.0]
        sigma = 2.0 + F.softplus(self.sigma_raw)

        # gamma in [0.3, 1.0]
        gamma = 0.3 + torch.sigmoid(self.gamma_raw) * 0.7

        # psi in [0, 2pi)
        psi = torch.sigmoid(self.psi_raw) * (2.0 * math.pi)

        # amplitude in [0.5, 1.5] (stable)
        amp = 0.5 + torch.sigmoid(self.amp_raw)

        kernels = []
        for i in range(self.n_filters):
            th = theta[i]
            fr = freq[i]
            sg = sigma[i]
            gm = gamma[i]
            ph = psi[i]

            # rotate coords
            x_prime = x * torch.cos(th) + y * torch.sin(th)
            y_prime = -x * torch.sin(th) + y * torch.cos(th)

            gauss = torch.exp(-0.5 * ((x_prime ** 2 + (gm ** 2) * (y_prime ** 2)) / (sg ** 2)))
            wave = torch.cos(2.0 * math.pi * fr * x_prime + ph)
            gabor = gauss * wave

            # zero-mean, unit-norm (stability)
            gabor = gabor - gabor.mean()
            gabor = gabor / (gabor.norm(p=2) + 1e-6)

            gabor = amp[i] * gabor
            kernels.append(gabor)

        k = torch.stack(kernels, dim=0)  # [N,ks,ks]
        return k.unsqueeze(1)            # [N,1,ks,ks]

    def forward(self, x_gray: torch.Tensor) -> torch.Tensor:
        if x_gray.ndim != 4 or x_gray.size(1) != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x_gray.shape)}")
        kernels = self._build_kernels(device=x_gray.device, dtype=x_gray.dtype)
        # same padding
        pad = self.ksize // 2
        x = F.pad(x_gray, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(x, kernels, bias=None, stride=1, padding=0)  # [B,N,H,W]
        return out


class GaborBranch(nn.Module):
    """
    Gray -> Gabor bank -> project to C channels -> refinement convs
    Output: [B,C,H,W]
    """
    def __init__(self, out_ch: int = 64, n_gabor: int = 32, ksize: int = 21):
        super().__init__()
        self.gabor = LearnableGaborBank(n_filters=n_gabor, ksize=ksize)
        self.proj = nn.Sequential(
            ConvBNAct(n_gabor, out_ch, k=1, s=1, p=0),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        xg = rgb_to_gray(x_rgb)                # [B,1,H,W]
        f = self.gabor(xg)                     # [B,n_gabor,H,W]
        f = self.proj(f)                       # [B,C,H,W]
        return f


# ---------------------------
# Branch 2: Spectral (DCT) Branch
# - Fixed DCT 8x8 filter bank, stride=8 -> coefficient maps -> upsample
# ---------------------------
def build_dct_basis_2d(N: int = 8, normalize: bool = True) -> torch.Tensor:
    """
    Returns DCT-II basis filters: [N*N, 1, N, N]
    """
    basis = []
    for u in range(N):
        for v in range(N):
            filt = torch.zeros((N, N), dtype=torch.float32)
            for x in range(N):
                for y in range(N):
                    cu = math.sqrt(1.0 / N) if u == 0 else math.sqrt(2.0 / N)
                    cv = math.sqrt(1.0 / N) if v == 0 else math.sqrt(2.0 / N)
                    filt[x, y] = cu * cv * math.cos((math.pi * (2 * x + 1) * u) / (2 * N)) * math.cos(
                        (math.pi * (2 * y + 1) * v) / (2 * N)
                    )
            basis.append(filt)
    B = torch.stack(basis, dim=0).unsqueeze(1)  # [64,1,8,8]

    if normalize:
        # unit norm per filter
        B = B / (B.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-6)
    return B


class SpectralDCTBranch(nn.Module):
    """
    Gray -> fixed DCT conv (stride 8) -> abs -> 1x1 mix -> upsample -> refine
    Output: [B,C,H,W]
    """
    def __init__(self, out_ch: int = 64, block: int = 8):
        super().__init__()
        self.block = block
        dct = build_dct_basis_2d(N=block, normalize=True)
        self.register_buffer("dct_kernels", dct, persistent=False)

        # Mix 64 coeff maps -> out_ch
        self.mix = nn.Sequential(
            ConvBNAct(block * block, out_ch, k=1, s=1, p=0),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        xg = rgb_to_gray(x_rgb)  # [B,1,H,W]
        # ensure size divisible by block via reflection pad
        B, C, H, W = xg.shape
        pad_h = (self.block - (H % self.block)) % self.block
        pad_w = (self.block - (W % self.block)) % self.block
        if pad_h != 0 or pad_w != 0:
            xg = F.pad(xg, (0, pad_w, 0, pad_h), mode="reflect")
        # stride=block gives block-wise DCT maps
        coef = F.conv2d(xg, self.dct_kernels.to(xg.device, xg.dtype), stride=self.block, padding=0)  # [B,64,H/8,W/8]
        coef = coef.abs()  # magnitude-like
        feat = self.mix(coef)  # [B,out_ch,h',w']

        # upsample back to original H,W
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return feat


# ---------------------------
# Branch 3: Edge + Soft Morphology
# - fixed edge filters + differentiable soft dilation/erosion
# ---------------------------
def fixed_edge_kernels() -> torch.Tensor:
    """
    Returns a small bank: SobelX, SobelY, Laplacian, ScharrX, ScharrY
    Shape: [5,1,3,3]
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32)
    lap = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32)
    scharr_x = torch.tensor([[3, 0, -3],
                             [10, 0, -10],
                             [3, 0, -3]], dtype=torch.float32)
    scharr_y = torch.tensor([[3, 10, 3],
                             [0, 0, 0],
                             [-3, -10, -3]], dtype=torch.float32)

    K = torch.stack([sobel_x, sobel_y, lap, scharr_x, scharr_y], dim=0).unsqueeze(1)
    # normalize per kernel
    K = K / (K.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-6)
    return K


def soft_dilate(x: torch.Tensor, k: int = 3, beta: float = 10.0) -> torch.Tensor:
    """
    Differentiable approximation of dilation using log-sum-exp over local window.
    """
    pad = k // 2
    xpad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    # unfold: [B,C, H*W, k*k]
    patches = xpad.unfold(2, k, 1).unfold(3, k, 1)  # [B,C,H,W,k,k]
    patches = patches.contiguous().view(*patches.shape[:4], -1)  # [B,C,H,W,k*k]
    # logsumexp over window
    return (1.0 / beta) * torch.logsumexp(beta * patches, dim=-1)


def soft_erode(x: torch.Tensor, k: int = 3, beta: float = 10.0) -> torch.Tensor:
    return -soft_dilate(-x, k=k, beta=beta)


class EdgeMorphBranch(nn.Module):
    """
    Gray -> fixed edge bank -> combine -> soft morphology -> project to C
    Output: [B,C,H,W]
    """
    def __init__(self, out_ch: int = 64, beta: float = 10.0):
        super().__init__()
        K = fixed_edge_kernels()
        self.register_buffer("edge_kernels", K, persistent=False)
        self.beta = beta

        # edge bank -> 5 channels, then to out_ch
        self.proj = nn.Sequential(
            ConvBNAct(5 * 3, out_ch, k=1, s=1, p=0),  # (edge + dilate + erode) concatenated
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        xg = rgb_to_gray(x_rgb)  # [B,1,H,W]
        pad = 1
        xpad = F.pad(xg, (pad, pad, pad, pad), mode="reflect")
        edges = F.conv2d(xpad, self.edge_kernels.to(xg.device, xg.dtype), stride=1, padding=0)  # [B,5,H,W]
        edges = torch.tanh(edges)  # stabilize

        dil = soft_dilate(edges, k=3, beta=self.beta)
        ero = soft_erode(edges, k=3, beta=self.beta)

        feat = torch.cat([edges, dil, ero], dim=1)  # [B,15,H,W]
        feat = self.proj(feat)                     # [B,C,H,W]
        return feat


# ---------------------------
# Router + Fusion
# ---------------------------
class BranchRouter(nn.Module):
    """
    Takes pooled features from each branch and outputs soft weights per sample.
    """
    def __init__(self, ch: int, hidden: int = 128, n_branches: int = 3):
        super().__init__()
        self.n_branches = n_branches
        self.mlp = nn.Sequential(
            nn.Linear(ch * n_branches, hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, n_branches),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats list each: [B,C,H,W]
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]  # each [B,C]
        x = torch.cat(pooled, dim=1)                                      # [B,3C]
        w = self.mlp(x)                                                   # [B,3]
        w = F.softmax(w, dim=1)
        return w


class FiLMoSNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        base_ch: int = 64,
        gabor_filters: int = 32,
        gabor_ksize: int = 21,
        dct_block: int = 8,
        morph_beta: float = 10.0,
    ):
        super().__init__()

        self.branch_gabor = GaborBranch(out_ch=base_ch, n_gabor=gabor_filters, ksize=gabor_ksize)
        self.branch_spec  = SpectralDCTBranch(out_ch=base_ch, block=dct_block)
        self.branch_morph = EdgeMorphBranch(out_ch=base_ch, beta=morph_beta)

        self.router = BranchRouter(ch=base_ch, hidden=128, n_branches=3)

        # post-fusion refinement (small backbone)
        self.refine = nn.Sequential(
            ConvBNAct(base_ch, base_ch, k=3, s=1),
            ConvBNAct(base_ch, base_ch * 2, k=3, s=2),  # downsample
            ConvBNAct(base_ch * 2, base_ch * 2, k=3, s=1),
            ConvBNAct(base_ch * 2, base_ch * 4, k=3, s=2),  # downsample
            ConvBNAct(base_ch * 4, base_ch * 4, k=3, s=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
            nn.Linear(base_ch * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict for transparency:
          - logits: [B,3]
          - router_w: [B,3]
        """
        f1 = self.branch_gabor(x)
        f2 = self.branch_spec(x)
        f3 = self.branch_morph(x)

        w = self.router([f1, f2, f3])  # [B,3]
        # weighted fusion: sum_i w_i * f_i
        # reshape weights for broadcasting
        w1 = w[:, 0].view(-1, 1, 1, 1)
        w2 = w[:, 1].view(-1, 1, 1, 1)
        w3 = w[:, 2].view(-1, 1, 1, 1)
        fused = w1 * f1 + w2 * f2 + w3 * f3  # [B,C,H,W]

        z = self.refine(fused)
        logits = self.head(z)

        return {"logits": logits, "router_w": w}

# ============================================================
# Strong transfer-learning baseline for small BUSI datasets
# ============================================================
class PretrainedClassifier(nn.Module):
    def __init__(self, model_name: str = "convnext_tiny", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        if self.model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif self.model_name == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif self.model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("model_name must be 'convnext_tiny', 'convnext_small', or 'efficientnet_b0'")

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

        # Always keep the classification head trainable.
        head = self.backbone.classifier
        for p in head.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.backbone(x)
        router_w = torch.zeros((x.size(0), 3), device=x.device, dtype=logits.dtype)
        return {"logits": logits, "router_w": router_w}


def build_model(cfg: Any, num_classes: int) -> nn.Module:
    if cfg.model_name.lower() == "filmos":
        return FiLMoSNet(
            num_classes=num_classes,
            base_ch=64,
            gabor_filters=32,
            gabor_ksize=21,
            dct_block=8,
            morph_beta=10.0,
        )

    return PretrainedClassifier(
        model_name=cfg.model_name,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
    )
