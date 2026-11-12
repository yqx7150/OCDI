from __future__ import annotations

import math

import numpy as np
import torch

from ocdi.ops.complex import to_complex

def ifft2c_torch_matlab(Kc: torch.Tensor) -> torch.Tensor:
    X = torch.fft.ifftshift(Kc, dim=(-2, -1))
    x = torch.fft.ifft2(X, dim=(-2, -1))
    return torch.fft.fftshift(x, dim=(-2, -1))


def k2img_sos_from_2cchw_matlab(K_2c: torch.Tensor) -> torch.Tensor:
    squeeze_b = K_2c.ndim == 3
    if squeeze_b:
        K_2c = K_2c.unsqueeze(0)
    Xc = to_complex(K_2c)
    img_c = ifft2c_torch_matlab(Xc)
    mag = torch.sqrt(torch.sum(torch.real(img_c) ** 2 + torch.imag(img_c) ** 2, dim=1))
    return mag.squeeze(0) if squeeze_b else mag


def roger_psnr_ssim_matlab_equiv(gt_mag: np.ndarray, pr_mag: np.ndarray, norm_mode: str='max', match_gain: bool=True, do_clip: bool=True, epsv: float=1e-12, ssim_ks: int=11, ssim_sigma: float=1.5):
    from scipy.signal import convolve2d
    g = np.abs(gt_mag).astype(np.float64)
    p = np.abs(pr_mag).astype(np.float64)
    scale = float(np.max(g)) if norm_mode == 'max' else float(np.percentile(g, 99))
    scale = max(scale, 1e-12)
    g01 = g / scale
    p01 = p / scale
    if match_gain:
        g01c_for_gain = np.clip(g01, 0.0, 1.0)
        a = float(np.dot(g01c_for_gain.reshape(-1), p01.reshape(-1))) / float(np.dot(p01.reshape(-1), p01.reshape(-1)) + epsv)
        p01 = p01 * a
    if do_clip:
        g01c = np.clip(g01, 0.0, 1.0)
        p01c = np.clip(p01, 0.0, 1.0)
    else:
        g01c, p01c = (g01, p01)
    mse = float(np.mean((g01c - p01c) ** 2))
    psnr = float(20.0 * np.log10(1.0 / (math.sqrt(mse) + epsv)))
    ax = np.arange(ssim_ks, dtype=np.float64) - (ssim_ks - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * ssim_sigma ** 2))
    k /= np.sum(k)
    mu1 = convolve2d(g01c, k, mode='same', boundary='symm')
    mu2 = convolve2d(p01c, k, mode='same', boundary='symm')
    mu1_sq, mu2_sq, mu1_mu2 = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
    sigma1_sq = convolve2d(g01c * g01c, k, mode='same', boundary='symm') - mu1_sq
    sigma2_sq = convolve2d(p01c * p01c, k, mode='same', boundary='symm') - mu2_sq
    sigma12 = convolve2d(g01c * p01c, k, mode='same', boundary='symm') - mu1_mu2
    C1, C2 = (0.01 ** 2, 0.03 ** 2)
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + epsv)
    return (psnr, float(np.mean(ssim_map)))
