from __future__ import annotations

import math

import numpy as np
import torch

def _make_phase_ramp(W: int, shift_frac: float) -> np.ndarray:
    m = np.arange(W, dtype=np.float64)
    center = W // 2
    ky = m - center
    return np.exp(-1j * 2.0 * np.pi * float(shift_frac) * ky)


def apply_shift_ky_np(K: np.ndarray, shift_frac: float) -> np.ndarray:
    if abs(float(shift_frac)) < 1e-12:
        return K
    H, W, C = K.shape
    ramp = _make_phase_ramp(W, shift_frac).reshape(1, W, 1)
    return (K * ramp).astype(np.complex64)


def apply_fov_shift_2cchw(K_2c: torch.Tensor, shift_frac: float, axis: str='x') -> torch.Tensor:
    if abs(float(shift_frac)) < 1e-12:
        return K_2c
    B, CC2, H, W = K_2c.shape
    C = CC2 // 2
    xr, xi = (K_2c[:, :C], K_2c[:, C:])
    if axis.lower() == 'x':
        m = torch.arange(W, device=K_2c.device, dtype=torch.float32)
        k = m - float(W // 2)
        phi = 2.0 * math.pi * float(shift_frac) * k
        co = torch.cos(phi).view(1, 1, 1, W)
        si = torch.sin(phi).view(1, 1, 1, W)
    elif axis.lower() == 'y':
        m = torch.arange(H, device=K_2c.device, dtype=torch.float32)
        k = m - float(H // 2)
        phi = 2.0 * math.pi * float(shift_frac) * k
        co = torch.cos(phi).view(1, 1, H, 1)
        si = torch.sin(phi).view(1, 1, H, 1)
    else:
        raise ValueError('--shift_axis must be x or y')
    yr = xr * co + xi * si
    yi = xi * co - xr * si
    return torch.cat([yr, yi], dim=1)
