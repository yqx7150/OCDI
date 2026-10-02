from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.ioff()

def tensor_to_gray01(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy().astype('float32')
    p = np.percentile(arr, 99.9) if np.isfinite(arr).all() else 0.0
    return np.clip(arr / p, 0, 1) if p > 0 else (arr - arr.min()) / (arr.max() - arr.min() + 1e-08)


def save_grid(arr_list, titles, cols, out_path, dpi=180):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = len(arr_list)
    rows = (N + cols - 1) // cols
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i, a in enumerate(arr_list):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(a, cmap='gray', vmin=0, vmax=1)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def _finite_percentile(arr: np.ndarray, q: float) -> float:
    a = arr.astype(np.float32).ravel()
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else 0.0


def get_panel_scale_from_gt(gt: np.ndarray, mode: str, p: float):
    if mode == 'gt_max':
        a = gt[np.isfinite(gt)]
        return float(a.max()) if a.size else 0.0
    if mode == 'gt_p99':
        return _finite_percentile(gt, p)
    return None


def save_clean_panel(img01: np.ndarray, out_path: str, text: str='', dpi: int=300, add_text: bool=True, text_color: str='#ffd400', text_size: int=10):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = plt.gca()
    ax.imshow(img01, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if add_text and text:
        ax.text(0.02, 0.05, text, transform=ax.transAxes, color=text_color, fontsize=text_size, ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.5))
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_error_panel(err: np.ndarray, out_path: str, vmin: float=0.0, vmax: float=0.01, dpi: int=300, cmap: str='viridis', add_cbar: bool=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = plt.gca()
    im = ax.imshow(err.astype(np.float32), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if add_cbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def array_to_gray01(img: np.ndarray, scale=None, p: float=99.0) -> np.ndarray:
    arr = img.astype(np.float32)
    if scale is None:
        scale = _finite_percentile(arr, p)
    if scale and scale > 0:
        return np.clip(arr / float(scale), 0.0, 1.0)
    mn, mx = (float(np.nanmin(arr)), float(np.nanmax(arr)))
    return np.clip((arr - mn) / (mx - mn + 1e-08), 0.0, 1.0)

_to_gray01 = tensor_to_gray01
