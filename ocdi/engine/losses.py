from __future__ import annotations

import torch
import torch.nn.functional as F

def ms_l1(img_pred, img_gt):
    l = F.l1_loss(img_pred, img_gt)
    d2p = F.avg_pool2d(img_pred.unsqueeze(1), 2, 2).squeeze(1)
    d2g = F.avg_pool2d(img_gt.unsqueeze(1), 2, 2).squeeze(1)
    l += 0.5 * F.l1_loss(d2p, d2g)
    d4p = F.avg_pool2d(d2p.unsqueeze(1), 2, 2).squeeze(1)
    d4g = F.avg_pool2d(d2g.unsqueeze(1), 2, 2).squeeze(1)
    l += 0.25 * F.l1_loss(d4p, d4g)
    return l


def low_high_weight(H, W, frac=0.25, w_low=3.0, device='cpu'):
    wy = int(W * frac / 2)
    wx = int(H * frac / 2)
    w = torch.ones(1, 1, H, W, device=device)
    w[:, :, H // 2 - wx:H // 2 + wx, W // 2 - wy:W // 2 + wy] = w_low
    return w


@torch.no_grad()
def build_bg_mask(img_mb: torch.Tensor, thr: float=0.88, dilate: int=5) -> torch.Tensor:
    """
    img_mb: [B, H, W], SoS magnitude of the input MB (or x_t)
    Returns bg_mask: [B, 1, H, W] with 1 on background, 0 on (dilated) foreground
    """
    assert img_mb.ndim == 3
    B, H, W = img_mb.shape
    vals = img_mb.view(B, -1)
    q = torch.quantile(vals, 0.999, dim=1, keepdim=True) + 1e-06
    img_n = (img_mb / q.view(B, 1, 1)).clamp_(0, 1)
    fg = (img_n > thr).float().unsqueeze(1)
    if dilate > 0:
        k = 2 * dilate + 1
        fg = F.max_pool2d(fg, kernel_size=k, stride=1, padding=dilate)
    bg = 1.0 - fg
    return bg


def bg_losses(img_pred: torch.Tensor, img_mb: torch.Tensor, thr: float, dilate: int, hpf_w: float) -> tuple:
    """
    img_pred: [B, H, W] SoS magnitude of predicted K1
    img_mb:   [B, H, W] SoS magnitude used to build the mask (e.g., x_t)
    Returns: (L_bg_img, L_bg_hpf, L_bg_total)
    """
    bg = build_bg_mask(img_mb.detach(), thr=thr, dilate=dilate)
    x = img_pred.unsqueeze(1)
    L_bg_img = torch.mean(bg * torch.abs(x))
    lp = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    hp = torch.abs(x - lp)
    L_bg_hpf = torch.mean(bg * hp)
    L_bg = L_bg_img + hpf_w * L_bg_hpf
    return (L_bg_img, L_bg_hpf, L_bg)
