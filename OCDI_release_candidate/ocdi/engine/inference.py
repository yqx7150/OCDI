from __future__ import annotations

import math

import numpy as np
import torch

from ocdi.ops.complex import hard_dc, soft_dc

@torch.no_grad()
def infer_mb_to_k1(net, K_sms: torch.Tensor, T=64, alpha_sched='cos', mask=None, use_hard_dc=False):
    net.eval()
    if isinstance(K_sms, np.ndarray):
        K_sms = torch.from_numpy(K_sms).float()
    if K_sms.ndim == 3:
        K_sms = K_sms.unsqueeze(0)
    x = K_sms.clone().to(next(net.parameters()).device)
    y_meas = K_sms.clone().to(next(net.parameters()).device)
    alpha_fn = (lambda t: t / float(T)) if alpha_sched == 'lin' else lambda t: math.sin(0.5 * math.pi * t / T)
    for t in range(T, 0, -1):
        t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
        delta = net(x, t_tensor)
        K1_hat = x - delta
        if t > 1:
            a_t, a_prev = (alpha_fn(t), alpha_fn(t - 1))
            K23_est = x - K1_hat
            x = K1_hat + a_prev / a_t * K23_est
            if mask is not None:
                lam_t = 0.8 * (t / T)
                x = hard_dc(x, y_meas, mask) if use_hard_dc else soft_dc(x, y_meas, mask=mask, lam=lam_t)
        else:
            x = K1_hat
    return x
