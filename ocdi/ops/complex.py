from __future__ import annotations

import numpy as np
import torch

def ifft2c_torch(Kc: torch.Tensor) -> torch.Tensor:
    X = torch.fft.ifftshift(Kc, dim=(-2, -1))
    x = torch.fft.ifft2(X, dim=(-2, -1), norm='ortho')
    return torch.fft.fftshift(x, dim=(-2, -1))


def to_complex(X_ri: torch.Tensor) -> torch.Tensor:
    squeeze_b = False
    if X_ri.ndim == 3:
        X_ri = X_ri.unsqueeze(0)
        squeeze_b = True
    B, CC2, H, W = X_ri.shape
    C = CC2 // 2
    Xr, Xi = (X_ri[:, :C], X_ri[:, C:])
    X_c = torch.complex(Xr, Xi)
    return X_c.squeeze(0) if squeeze_b else X_c


def from_complex(X_c: torch.Tensor) -> torch.Tensor:
    squeeze_b = False
    if X_c.ndim == 3:
        X_c = X_c.unsqueeze(0)
        squeeze_b = True
    Xr, Xi = (torch.real(X_c), torch.imag(X_c))
    X_ri = torch.cat([Xr, Xi], dim=1)
    return X_ri.squeeze(0) if squeeze_b else X_ri


def complex_abs(X_ri: torch.Tensor) -> torch.Tensor:
    return torch.abs(to_complex(X_ri))


def k2img_sos_from_2cchw(K_2c: torch.Tensor) -> torch.Tensor:
    squeeze_b = K_2c.ndim == 3
    if squeeze_b:
        K_2c = K_2c.unsqueeze(0)
    Xc = to_complex(K_2c)
    img_c = ifft2c_torch(Xc)
    mag = torch.sqrt(torch.sum(torch.real(img_c) ** 2 + torch.imag(img_c) ** 2, dim=1))
    return mag.squeeze(0) if squeeze_b else mag


def forward_to_mb(K1: torch.Tensor, K23: torch.Tensor, alpha=1.0, mask=None) -> torch.Tensor:
    y_hat = K1 + alpha * K23
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(K1.device).view(1, 1, mask.shape[0], mask.shape[1])
        else:
            mask_t = mask.clone().to(K1.device)
            if mask_t.ndim == 2:
                mask_t = mask_t.view(1, 1, mask_t.shape[0], mask_t.shape[1])
            elif mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0)
        B, CC2, H, W = K1.shape
        y_hat = y_hat * mask_t.expand(B, 1, H, W).repeat(1, CC2, 1, 1)
    return y_hat


def hard_dc(pred: torch.Tensor, meas: torch.Tensor, mask=None) -> torch.Tensor:
    if mask is None:
        return meas.clone()
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(pred.device).view(1, 1, mask.shape[0], mask.shape[1])
    else:
        mask_t = mask.clone().to(pred.device)
        if mask_t.ndim == 2:
            mask_t = mask_t.view(1, 1, mask_t.shape[0], mask_t.shape[1])
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
    B, CC2, H, W = pred.shape
    M = mask_t.expand(B, 1, H, W).repeat(1, CC2, 1, 1)
    return M * meas + (1 - M) * pred


def soft_dc(pred: torch.Tensor, meas: torch.Tensor, mask=None, lam=0.1) -> torch.Tensor:
    if mask is None:
        return lam * meas + (1 - lam) * pred
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(pred.device).view(1, 1, mask.shape[0], mask.shape[1])
    else:
        mask_t = mask.clone().to(pred.device)
        if mask_t.ndim == 2:
            mask_t = mask_t.view(1, 1, mask_t.shape[0], mask_t.shape[1])
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
    B, CC2, H, W = pred.shape
    M = mask_t.expand(B, 1, H, W).repeat(1, CC2, 1, 1)
    return M * (lam * meas + (1 - lam) * pred) + (1 - M) * pred


def to_2cchw(x_hwC: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x_hwC.astype(np.complex64)).permute(2, 0, 1)
    return torch.cat([torch.real(x), torch.imag(x)], dim=0).float()


def complex_abs_2cchw(x_2c: torch.Tensor) -> torch.Tensor:
    C2 = x_2c.shape[0]
    C = C2 // 2
    return torch.sqrt(x_2c[:C] ** 2 + x_2c[C:] ** 2 + 1e-12)


def quantile_scale_kmb(KMB_2c: torch.Tensor, q: float = 0.999) -> float:
    vals = complex_abs_2cchw(KMB_2c).reshape(-1)
    return float(torch.quantile(vals, q).item() + 1e-6)
