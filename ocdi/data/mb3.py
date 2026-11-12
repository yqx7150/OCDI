"""MB3 training dataset restored from the supplied MB3R1.PY.

The training interpolation, normalization, sampling and optional K23 scaling
follow that source. I/O is explicit HWC; silent axis guessing is removed.
Legacy coil truncation/padding is retained ONLY for this training interface.
"""
from __future__ import annotations
import glob
import math
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .packed import load_packed_mb3_mat, load_mat_arrays, _field
from ocdi.ops.complex import to_2cchw, complex_abs


def center_crop_or_pad_np(array, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Target dimensions must be positive")
    h, w, coils = array.shape
    if h > height:
        offset = (h - height) // 2
        array = array[offset:offset + height]
    if w > width:
        offset = (w - width) // 2
        array = array[:, offset:offset + width]
    h, w, _ = array.shape
    output = np.zeros((height, width, coils), dtype=array.dtype)
    top, left = (height - h) // 2, (width - w) // 2
    output[top:top + h, left:left + w] = array
    return output


class MB3FolderDataset(Dataset):
    def __init__(self, root_dir, T=64, alpha_sched="cos", amp_norm=True,
                 amp_norm_mode="kmb", sum23_div_sqrt2=False, target_hw=None,
                 coils=12, tsample="uniform"):
        self.root_dir = root_dir
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.mat")))
        if not self.paths:
            raise ValueError(f"No .mat found in {root_dir}")
        if T <= 0 or alpha_sched not in ("lin", "cos") or tsample not in ("uniform", "mid"):
            raise ValueError("Invalid interpolation or timestep settings")
        if amp_norm_mode not in ("k1", "kmb"):
            raise ValueError("Invalid normalization mode")
        self.T, self.alpha_sched, self.amp_norm = T, alpha_sched, amp_norm
        self.amp_norm_mode, self.sum23_div_sqrt2 = amp_norm_mode, sum23_div_sqrt2
        self.target_hw, self.coils, self.tsample = target_hw, coils, tsample

    def __len__(self):
        return len(self.paths)

    def alpha(self, t):
        return t / float(self.T) if self.alpha_sched == "lin" else math.sin(0.5 * math.pi * t / self.T)

    def _sample_t(self):
        if self.tsample == "uniform":
            return random.randint(1, self.T)
        u = random.random()
        return min(max(1 + int((self.T - 1) * math.sin(0.5 * math.pi * u) ** 2), 1), self.T)

    def __getitem__(self, index):
        path = self.paths[index]
        k1, k2, k3, kmb = load_packed_mb3_mat(path, self.coils, coil_policy="legacy")
        # Preserve the original fallback when optional K23/sqrt(2) is enabled.
        has_file_kmb = _field(load_mat_arrays(path), "KMB", required=False) is not None
        if self.target_hw is not None:
            k1, k2, k3, kmb = [center_crop_or_pad_np(x, *self.target_hw) for x in (k1, k2, k3, kmb)]
        k23 = k2 + k3
        if self.sum23_div_sqrt2:
            k23 = k23 / np.float32(np.sqrt(2.0))
        k1, k23 = to_2cchw(k1), to_2cchw(k23)
        kmb = to_2cchw(kmb) if has_file_kmb else k1 + k23
        if self.amp_norm:
            source = kmb if self.amp_norm_mode == "kmb" else k1
            scale = torch.quantile(complex_abs(source).reshape(-1), 0.999) + 1e-6
            k1, k23, kmb = k1 / scale, k23 / scale, kmb / scale
        t = self._sample_t()
        return (k1 + self.alpha(t) * k23).float(), torch.tensor(t, dtype=torch.long), k1.float(), k23.float(), kmb.float()
