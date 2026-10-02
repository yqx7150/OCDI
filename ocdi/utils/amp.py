from __future__ import annotations

import torch

try:
    from torch import amp as torch_amp
    _NEW_AMP = True
except Exception:
    import torch.cuda.amp as torch_amp
    _NEW_AMP = False

def _amp_autocast(enabled=True, dtype=torch.float16, device='cuda'):
    if _NEW_AMP:
        return torch_amp.autocast(device, enabled=enabled, dtype=dtype)
    else:
        return torch_amp.autocast(enabled=enabled, dtype=dtype)


def _amp_scaler(device='cuda'):
    if _NEW_AMP:
        return torch_amp.GradScaler(device)
    else:
        return torch_amp.GradScaler()
