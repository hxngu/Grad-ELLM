"""Utility functions used by Grad-ELLM evaluation."""

import random

import numpy as np
import torch


def reset_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, min=1e-12)
    q = torch.clamp(q, min=1e-12)
    scale = torch.sqrt(torch.tensor(2.0, device=p.device, dtype=p.dtype))
    return torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=-1)) / scale


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    cosine_similarity = dot_product / (norm_x * norm_y + 1e-8)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    return 1 - cosine_similarity


def alpha_scaling(
    emap_slice: torch.Tensor,
    p: float,
    tol: float = 1e-4,
    max_iter: int = 60,
    *,
    alpha_min: float = 0.0,
    alpha_max: float = 128.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply the alpha transformation used by pi-Soft-NC/NS.

    Finds alpha such that mean(x ** alpha) approximately equals the requested
    expected retaining probability p.
    """
    if not torch.is_tensor(emap_slice):
        raise TypeError("emap_slice must be a torch.Tensor")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    x = emap_slice
    if x.numel() == 0:
        return x.clone()

    original_dtype = x.dtype
    x = torch.clamp(x.to(dtype=torch.float32), 0.0, 1.0)
    x_stable = torch.clamp(x, eps, 1.0)

    if p == 0.0:
        return torch.pow(x_stable, float(alpha_max)).to(original_dtype)
    if p == 1.0:
        return torch.ones_like(x, dtype=original_dtype)
    if torch.all(x <= 0.0):
        return torch.zeros_like(x, dtype=original_dtype)
    if torch.all(x >= 1.0):
        return torch.ones_like(x, dtype=original_dtype)

    def mean_after_scaling(alpha: float) -> float:
        return torch.pow(x_stable, alpha).mean().item()

    frac_ones = (x >= 1.0 - 1e-8).float().mean().item()
    if p < frac_ones - (10 * tol):
        return torch.pow(x_stable, float(alpha_max)).to(original_dtype)
    if 1.0 - p <= tol:
        return torch.ones_like(x, dtype=original_dtype)

    low, high = float(alpha_min), float(alpha_max)
    f_high = mean_after_scaling(high)
    expand_steps = 0
    while f_high > p and expand_steps < 20:
        high *= 2.0
        f_high = mean_after_scaling(high)
        expand_steps += 1

    if f_high > p:
        return torch.pow(x_stable, high).to(original_dtype)

    err_tol = max(float(tol), float(tol) * float(p))
    alpha = high
    for _ in range(int(max_iter)):
        mid = (low + high) / 2.0
        f_mid = mean_after_scaling(mid)
        alpha = mid
        if abs(f_mid - p) <= err_tol:
            break
        if f_mid > p:
            low = mid
        else:
            high = mid

    return torch.pow(x_stable, alpha).to(original_dtype)


def softmax_scaling(heatmap: torch.Tensor) -> torch.Tensor:
    return torch.softmax(heatmap, dim=-1)


def l1_scaling(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    abs_heatmap = torch.abs(heatmap)
    sum_abs = torch.sum(abs_heatmap)
    if sum_abs == 0:
        return heatmap
    return abs_heatmap / (sum_abs + eps)


def default_scaling(heatmap: torch.Tensor) -> torch.Tensor:
    return heatmap
