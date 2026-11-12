# -*- coding: utf-8 -*-
"""
MB3 training script — UPDEC + AUX-K23 版本
在 train_mb3_from10_8_twostream_xattn_fixpad_UPDEC.py 基础上：
1) Decoder 采用 A/B 双流上采样 + Cross-Interaction（同 UPDEC）。
2) 新增一个**轻量 K23 辅助头**（仅训练期使用）：从 B 流解码末端预测 k23_pred ≈ (K2+K3)。
3) 新增三项可选损失（flag 控制，默认全部关闭）：
   - 前向一致性 L_fc：\| (K1_pred + a_t·k23_pred) − x_t \|
   - K23 监督  L_k23：\| k23_pred − (K2+K3) \|
   - 正交惩罚 L_orth：mean(|cos_sim(K1_pred, k23_pred)|)
推理阶段仍只产主头 delta_k1；无需改任何下游推理代码。
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')

import re, random, math, glob, argparse
import numpy as np
import scipy.io as sio
try:
    import h5py
except Exception:
    h5py = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==== AMP helpers ====
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

plt.ioff()

# ==== complex & FFT helpers ====

def ifft2c_torch(Kc: torch.Tensor) -> torch.Tensor:
    X = torch.fft.ifftshift(Kc, dim=(-2,-1))
    x = torch.fft.ifft2(X, dim=(-2,-1), norm='ortho')
    return torch.fft.fftshift(x, dim=(-2,-1))

def to_complex(X_ri: torch.Tensor) -> torch.Tensor:
    squeeze_b = False
    if X_ri.ndim == 3:
        X_ri = X_ri.unsqueeze(0); squeeze_b = True
    B, CC2, H, W = X_ri.shape
    C = CC2 // 2
    Xr, Xi = X_ri[:, :C], X_ri[:, C:]
    X_c = torch.complex(Xr, Xi)
    return X_c.squeeze(0) if squeeze_b else X_c

def from_complex(X_c: torch.Tensor) -> torch.Tensor:
    squeeze_b = False
    if X_c.ndim == 3: X_c = X_c.unsqueeze(0); squeeze_b = True
    Xr, Xi = torch.real(X_c), torch.imag(X_c)
    X_ri = torch.cat([Xr, Xi], dim=1)
    return X_ri.squeeze(0) if squeeze_b else X_ri

def complex_abs(X_ri: torch.Tensor) -> torch.Tensor:
    return torch.abs(to_complex(X_ri))

def k2img_sos_from_2cchw(K_2c: torch.Tensor) -> torch.Tensor:
    squeeze_b = (K_2c.ndim == 3)
    if squeeze_b: K_2c = K_2c.unsqueeze(0)
    Xc = to_complex(K_2c)
    img_c = ifft2c_torch(Xc)
    mag = torch.sqrt(torch.sum(torch.real(img_c)**2 + torch.imag(img_c)**2, dim=1))
    return mag.squeeze(0) if squeeze_b else mag

def _to_gray01(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy().astype('float32')
    p = np.percentile(arr, 99.9) if np.isfinite(arr).all() else 0.0
    return np.clip(arr/p, 0, 1) if p > 0 else (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def save_grid(arr_list, titles, cols, out_path, dpi=180):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = len(arr_list); rows = (N + cols - 1) // cols
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, a in enumerate(arr_list):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(a, cmap='gray', vmin=0, vmax=1)
        if titles and i < len(titles): ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    plt.tight_layout(); plt.savefig(out_path, dpi=dpi); plt.close()

def _gn(ch, groups=8):
    g = groups if ch % groups == 0 and ch >= groups else 1
    return nn.GroupNorm(g, ch)

# ==== basic blocks ====
class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(ch // r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Conv2d(ch, hidden, 1, bias=True), nn.SiLU(),
                                 nn.Conv2d(hidden, ch, 1, bias=True), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.avg(x))

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128, T=64):
        super().__init__()
        self.emb = nn.Embedding(T+1, dim)
    def forward(self, t):
        return self.emb(t)

class FiLM(nn.Module):
    def __init__(self, time_dim, ch):
        super().__init__()
        self.mlp = nn.Linear(time_dim, ch*2)
        nn.init.zeros_(self.mlp.weight); nn.init.zeros_(self.mlp.bias)
    def forward(self, x, t_emb):
        g, b = torch.chunk(self.mlp(t_emb)[:, :, None, None], 2, dim=1)
        return x * (1 + g) + b

class LKRSEFiLM(nn.Module):
    def __init__(self, ch, time_dim, k=(3,9), res_scale=0.1):
        super().__init__()
        if isinstance(k, tuple):
            padding = (k[0]//2, k[1]//2)
            self.dw  = nn.Conv2d(ch, ch, k, padding=padding, groups=ch, bias=False)
        else:
            self.dw  = nn.Conv2d(ch, ch, k, padding=k//2, groups=ch, bias=False)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.gn1 = _gn(ch); self.gn2 = _gn(ch)
        self.act = nn.SiLU()
        self.film = FiLM(time_dim, ch)
        self.se   = SE(ch)
        self.res_scale = res_scale
    def forward(self, x, t_emb):
        idt = x
        y = self.dw(x); y = self.gn1(y); y = self.act(y)
        y = self.pw(y); y = self.gn2(y)
        y = self.film(y, t_emb); y = self.act(y)
        y = self.se(y)
        return idt + self.res_scale * y

class CrossInteractionBlock(nn.Module):
    def __init__(self, ch, time_dim=None):
        super().__init__()
        self.conv_mix = nn.Conv2d(ch*2, ch*2, 3, padding=1, bias=False)
        self.norm     = _gn(ch*2)
        self.act      = nn.SiLU()
        self.film     = FiLM(time_dim, ch*2) if time_dim is not None else None
        self.proj_a   = nn.Conv2d(ch*2, ch, 1, bias=False)
        self.proj_b   = nn.Conv2d(ch*2, ch, 1, bias=False)
    def forward(self, a, b, t_emb=None):
        h = torch.cat([a, b], dim=1)
        h = self.conv_mix(h); h = self.norm(h)
        if (self.film is not None) and (t_emb is not None):
            h = self.film(h, t_emb)
        h = self.act(h)
        g_a = self.proj_a(h); g_b = self.proj_b(h)
        return a + g_a, b + g_b

class UpsamplePS(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, mid_ch*4, 1, bias=False)
        self.ps  = nn.PixelShuffle(2)
        self.bn1 = _gn(mid_ch); self.bn2 = _gn(out_ch)
        self.conv= nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.act = nn.SiLU()
        with torch.no_grad():
            out_c, in_c, k1, k2 = self.pre.weight.shape
            subc = out_c // 4
            weight = torch.zeros(out_c, in_c, k1, k2)
            nn.init.kaiming_normal_(weight[:subc])
            weight = weight.view(subc, 4, in_c, k1, k2)
            weight = weight.permute(1,0,2,3,4).contiguous().view(out_c, in_c, k1, k2)
            self.pre.weight.copy_(weight)
    def forward(self, x):
        x = self.ps(self.pre(x))
        x = self.bn1(x); x = self.act(x)
        x = self.conv(x)
        x = self.bn2(x); x = self.act(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = _gn(out_ch)
        self.act  = nn.SiLU()
        self.film = FiLM(time_dim, out_ch)
    def forward(self, x, t_emb):
        x = self.conv(x); x = self.norm(x); x = self.film(x, t_emb); return self.act(x)

# ==== windowed attention utils（多头窗注意力，与基线一致）====

def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))

def window_partition(x, window_size):
    Wh, Ww = _to_2tuple(window_size)
    B, C, H0, W0 = x.shape
    pad_h = (Wh - H0 % Wh) % Wh
    pad_w = (Ww - W0 % Ww) % Ww
    H = H0 + pad_h; W = W0 + pad_w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    x = x.view(B, C, H//Wh, Wh, W//Ww, Ww)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, Wh*Ww, C)
    meta = (B, C, H0, W0, H, W, Wh, Ww)
    return windows, meta

def window_reverse(windows, meta):
    B, C, H0, W0, H, W, Wh, Ww = meta
    nH, nW = H//Wh, W//Ww
    x = windows.view(B, nH, nW, Wh, Ww, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    if (H != H0) or (W != W0):
        x = x[:, :, :H0, :W0]
    return x

def _make_divisible(x, m):
    if x % m == 0: return x
    y = x - (x % m)
    return max(y, m)

class WindowMHA(nn.Module):
    def __init__(self, ch, num_heads=4, window_size=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window = _to_2tuple(window_size)
        embed_dim = _make_divisible(ch, num_heads)
        self.in_proj  = nn.Conv2d(ch, embed_dim, 1, bias=True)
        self.qkv      = nn.Linear(embed_dim, embed_dim*3, bias=True)
        self.proj     = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Conv2d(embed_dim, ch, 1, bias=True)
        self.attn_drop= nn.Dropout(attn_drop)
        self.proj_drop= nn.Dropout(proj_drop)
        self.scale = (embed_dim // num_heads) ** -0.5
    def forward(self, x):
        B, C, H0, W0 = x.shape
        x_embed = self.in_proj(x)
        windows, meta = window_partition(x_embed, self.window)
        qkv = self.qkv(windows)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = q.shape[-1] // self.num_heads
        q = q.view(q.shape[0], q.shape[1], self.num_heads, head_dim).permute(0,2,1,3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, head_dim).permute(0,2,1,3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, head_dim).permute(0,2,1,3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0,2,1,3).contiguous().view(windows.shape[0], windows.shape[1], -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = window_reverse(out, meta)
        out = self.out_proj(out)
        return out

class WindowCrossAttn(nn.Module):
    def __init__(self, ch_q, ch_kv, num_heads=4, window_size=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window = _to_2tuple(window_size)
        embed_q = _make_divisible(ch_q, num_heads)
        embed_kv = _make_divisible(ch_kv, num_heads)
        self.proj_q  = nn.Conv2d(ch_q,  embed_q, 1, bias=True)
        self.proj_kv = nn.Conv2d(ch_kv, embed_kv, 1, bias=True)
        self.q_proj  = nn.Linear(embed_q,  embed_q, bias=True)
        self.kv_proj = nn.Linear(embed_kv, embed_kv*2, bias=True)
        self.proj    = nn.Linear(embed_q,  embed_q, bias=True)
        self.out     = nn.Conv2d(embed_q, ch_q, 1, bias=True)
        self.scale = (embed_q // num_heads) ** -0.5
        self.attn_drop= nn.Dropout(attn_drop)
        self.proj_drop= nn.Dropout(proj_drop)
    def forward(self, Qa, Kb):
        B, Ca, H0, W0 = Qa.shape
        _, Cb, Hb, Wb = Kb.shape
        assert H0==Hb and W0==Wb, "CrossAttn expects same spatial size for A/B"
        Qa_e = self.proj_q(Qa)
        Kb_e = self.proj_kv(Kb)
        Qwin, meta = window_partition(Qa_e, self.window)
        KVwin, _   = window_partition(Kb_e, self.window)
        q = self.q_proj(Qwin)
        kv = self.kv_proj(KVwin)
        k, v = kv.chunk(2, dim=-1)
        head_dim = q.shape[-1] // self.num_heads
        q = q.view(q.shape[0], q.shape[1], self.num_heads, head_dim).permute(0,2,1,3)
        head_dim_k = k.shape[-1] // self.num_heads
        k = k.view(k.shape[0], k.shape[1], self.num_heads, head_dim_k).permute(0,2,1,3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, head_dim_k).permute(0,2,1,3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0,2,1,3).contiguous().view(Qwin.shape[0], Qwin.shape[1], -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = window_reverse(out, meta)
        out = self.out(out)
        return out

class MLP2d(nn.Module):
    def __init__(self, ch, mlp_ratio=2.0):
        super().__init__()
        hidden = int(ch * mlp_ratio)
        self.fc1 = nn.Conv2d(ch, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, ch, 1, bias=True)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class DualStreamXAttnBlock(nn.Module):
    def __init__(self, ch, heads=4, window=6, mlp_ratio=2.0):
        super().__init__()
        self.sa_a = WindowMHA(ch, num_heads=heads, window_size=window)
        self.sa_b = WindowMHA(ch, num_heads=heads, window_size=window)
        self.ca_ab = WindowCrossAttn(ch, ch, num_heads=heads, window_size=window)
        self.ca_ba = WindowCrossAttn(ch, ch, num_heads=heads, window_size=window)
        self.norm_a1 = _gn(ch); self.norm_b1 = _gn(ch)
        self.norm_a2 = _gn(ch); self.norm_b2 = _gn(ch)
        self.mlp_a = MLP2d(ch, mlp_ratio=mlp_ratio)
        self.mlp_b = MLP2d(ch, mlp_ratio=mlp_ratio)
    def forward(self, a, b):
        a1 = a + self.sa_a(self.norm_a1(a))
        b1 = b + self.sa_b(self.norm_b1(b))
        a2 = a1 + self.ca_ab(self.norm_a2(a1), self.norm_b2(b1))
        b2 = b1 + self.ca_ba(self.norm_b2(b1), self.norm_a2(a1))
        return a2 + self.mlp_a(a2), b2 + self.mlp_b(b2)

# ==== Dataset / helpers / EMA / losses ====

def _load_mat_with_fallback(path: str):
    try:
        md = sio.loadmat(path)
        return md, True
    except Exception:
        if h5py is None:
            raise
        with h5py.File(path, 'r') as f:
            md = {k: f[k][()] for k in f.keys()}
        return md, False

def _ensure_hwcoils_np(K: np.ndarray) -> np.ndarray:
    if K.ndim == 2:
        K = K[..., None]
    if K.shape[0] in (8,12,16,20) and K.ndim == 3:
        K = K.transpose(1,2,0)
    elif K.shape[-1] in (8,12,16,20):
        pass
    else:
        K = np.moveaxis(K, int(np.argmin(K.shape)), -1)
    return K

def enforce_coils_np(K: np.ndarray, target_C: int) -> np.ndarray:
    H, W, C0 = K.shape
    if C0 >= target_C:
        return K[..., :target_C]
    pad = np.zeros((H, W, target_C - C0), dtype=K.dtype)
    return np.concatenate([K, pad], axis=2)

def center_crop_or_pad_np(K: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    H, W, C = K.shape
    if H > Ht: dh = (H - Ht)//2; K = K[dh:dh+Ht, :, :]
    if W > Wt: dw = (W - Wt)//2; K = K[:, dw:dw+Wt, :]
    H2, W2, _ = K.shape
    if H2 < Ht or W2 < Wt:
        K2 = np.zeros((Ht, Wt, C), dtype=K.dtype)
        h_off = (Ht - H2)//2; w_off = (Wt - W2)//2
        K2[h_off:h_off+H2, w_off:w_off+W2, :] = K; K = K2
    return K

class MB3FolderDataset(Dataset):
    def __init__(self, root_dir: str, T=64, alpha_sched='cos', amp_norm=True,
                 amp_norm_mode='kmb', sum23_div_sqrt2=False,
                 target_hw=None, coils=12, tsample='uniform'):
        super().__init__()
        self.root_dir = root_dir
        self.paths = sorted([p for p in glob.glob(os.path.join(root_dir, '*.mat'))])
        assert len(self.paths) > 0, f'No .mat found in {root_dir}'
        self.T = T; self.alpha_sched = alpha_sched; self.amp_norm = amp_norm
        self.amp_norm_mode = amp_norm_mode
        self.sum23_div_sqrt2 = sum23_div_sqrt2
        self.target_hw = target_hw; self.coils = coils; self.tsample = tsample
    def __len__(self):
        return len(self.paths)
    def alpha(self, t):
        return t/float(self.T) if self.alpha_sched=='lin' else math.sin(0.5*math.pi*t/self.T)
    def _sample_t(self):
        if self.tsample == 'uniform':
            return random.randint(1, self.T)
        u = random.random(); t = 1 + int((self.T - 1) * (math.sin(0.5*math.pi*u) ** 2))
        return min(max(t,1), self.T)
    def __getitem__(self, idx):
        path = self.paths[idx]
        md, _ = _load_mat_with_fallback(path)
        def get_arr(key):
            if key in md: return md[key]
            for k in md.keys():
                if str(k).strip().lower() == key.lower():
                    return md[k]
            raise KeyError(f'{key} not found in {path}')
        K1 = np.asarray(get_arr('K1'))
        K2 = np.asarray(get_arr('K2'))
        K3 = np.asarray(get_arr('K3'))
        KMB_file = None
        if 'KMB' in md or 'kmb' in md:
            try:
                KMB_file = np.asarray(get_arr('KMB'))
            except Exception:
                try: KMB_file = np.asarray(get_arr('kmb'))
                except Exception: KMB_file = None
        def to_c64_np(x):
            if (not np.iscomplexobj(x)) and x.dtype.kind in ('V','O') and h5py is not None and x.shape[-1] == 2:
                x = x[...,0] + 1j*x[...,1]
            return x.astype(np.complex64) if np.iscomplexobj(x) else (x.astype(np.float32) + 0j).astype(np.complex64)
        K1 = to_c64_np(_ensure_hwcoils_np(K1))
        K2 = to_c64_np(_ensure_hwcoils_np(K2))
        K3 = to_c64_np(_ensure_hwcoils_np(K3))
        if KMB_file is not None:
            KMB_file = to_c64_np(_ensure_hwcoils_np(KMB_file))
        if self.target_hw is not None:
            Ht, Wt = self.target_hw
            K1 = center_crop_or_pad_np(K1, Ht, Wt)
            K2 = center_crop_or_pad_np(K2, Ht, Wt)
            K3 = center_crop_or_pad_np(K3, Ht, Wt)
            if KMB_file is not None:
                KMB_file = center_crop_or_pad_np(KMB_file, Ht, Wt)
        K1 = enforce_coils_np(K1, self.coils)
        K2 = enforce_coils_np(K2, self.coils)
        K3 = enforce_coils_np(K3, self.coils)
        if KMB_file is not None:
            KMB_file = enforce_coils_np(KMB_file, self.coils)
        K2sum = K2 + K3
        if self.sum23_div_sqrt2:
            K2sum = K2sum / np.sqrt(2.0).astype(np.float32)
        def to_2cchw(x_np):
            x = torch.from_numpy(x_np.astype(np.complex64)).permute(2,0,1)
            return torch.cat([torch.real(x), torch.imag(x)], dim=0)
        K1_t   = to_2cchw(K1)
        K23_t  = to_2cchw(K2sum)
        KMB_t = to_2cchw(KMB_file) if (KMB_file is not None) else (K1_t + K23_t)
        if self.amp_norm:
            if (self.amp_norm_mode.lower() == 'kmb') and (KMB_t is not None):
                scale = torch.quantile(complex_abs(KMB_t).reshape(-1), 0.999) + 1e-6
            elif self.amp_norm_mode.lower() == 'kmb':
                scale = torch.quantile(complex_abs(K1_t + K23_t).reshape(-1), 0.999) + 1e-6
            else:
                scale = torch.quantile(complex_abs(K1_t).reshape(-1), 0.999) + 1e-6
            K1_t  = K1_t  / scale
            K23_t = K23_t / scale
            if KMB_t is not None:
                KMB_t = KMB_t / scale
        t_idx = self._sample_t()
        a = self.alpha(t_idx)
        x_t = K1_t + (a * K23_t)
        if KMB_t is None:
            return x_t.float(), torch.tensor(t_idx, dtype=torch.long), K1_t.float(), K23_t.float()
        else:
            return x_t.float(), torch.tensor(t_idx, dtype=torch.long), K1_t.float(), K23_t.float(), KMB_t.float()

def forward_to_mb(K1: torch.Tensor, K23: torch.Tensor, alpha=1.0, mask=None) -> torch.Tensor:
    y_hat = K1 + (alpha * K23)
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(K1.device).view(1,1,mask.shape[0], mask.shape[1])
        else:
            mask_t = mask.clone().to(K1.device)
            if mask_t.ndim == 2: mask_t = mask_t.view(1,1,mask_t.shape[0], mask_t.shape[1])
            elif mask_t.ndim == 3: mask_t = mask_t.unsqueeze(0)
        B, CC2, H, W = K1.shape
        y_hat = y_hat * mask_t.expand(B, 1, H, W).repeat(1, CC2, 1, 1)
    return y_hat

def hard_dc(pred: torch.Tensor, meas: torch.Tensor, mask=None) -> torch.Tensor:
    if mask is None:
        return meas.clone()
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(pred.device).view(1,1,mask.shape[0], mask.shape[1])
    else:
        mask_t = mask.clone().to(pred.device)
        if mask_t.ndim == 2: mask_t = mask_t.view(1,1,mask_t.shape[0], mask_t.shape[1])
        elif mask_t.ndim == 3: mask_t = mask_t.unsqueeze(0)
    B, CC2, H, W = pred.shape
    M = mask_t.expand(B,1,H,W).repeat(1,CC2,1,1)
    return M * meas + (1 - M) * pred

def soft_dc(pred: torch.Tensor, meas: torch.Tensor, mask=None, lam=0.1) -> torch.Tensor:
    if mask is None:
        return lam * meas + (1 - lam) * pred
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(pred.device).view(1,1,mask.shape[0], mask.shape[1])
    else:
        mask_t = mask.clone().to(pred.device)
        if mask_t.ndim == 2:
            mask_t = mask_t.view(1,1,mask_t.shape[0], mask_t.shape[1])
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
    B, CC2, H, W = pred.shape
    M = mask_t.expand(B,1,H,W).repeat(1,CC2,1,1)
    return M * (lam*meas + (1-lam)*pred) + (1 - M) * pred

@torch.no_grad()
def infer_mb_to_k1(net, K_sms: torch.Tensor, T=64, alpha_sched='cos', mask=None, use_hard_dc=False):
    net.eval()
    if isinstance(K_sms, np.ndarray): K_sms = torch.from_numpy(K_sms).float()
    if K_sms.ndim == 3: K_sms = K_sms.unsqueeze(0)
    x = K_sms.clone().to(next(net.parameters()).device)
    y_meas = K_sms.clone().to(next(net.parameters()).device)
    alpha_fn = (lambda t: t/float(T)) if alpha_sched=='lin' else (lambda t: math.sin(0.5*math.pi*t/T))
    for t in range(T, 0, -1):
        t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
        delta = net(x, t_tensor)  # 推理阶段主输出仍为 delta
        K1_hat = x - delta
        if t > 1:
            a_t, a_prev = alpha_fn(t), alpha_fn(t-1)
            K23_est = x - K1_hat
            x = K1_hat + (a_prev / a_t) * K23_est
            if mask is not None:
                lam_t = 0.8 * (t / T)
                x = hard_dc(x, y_meas, mask) if use_hard_dc else soft_dc(x, y_meas, mask=mask, lam=lam_t)
        else:
            x = K1_hat
    return x

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        for n,p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)

def ms_l1(img_pred, img_gt):
    l = F.l1_loss(img_pred, img_gt)
    d2p = F.avg_pool2d(img_pred.unsqueeze(1), 2, 2).squeeze(1)
    d2g = F.avg_pool2d(img_gt  .unsqueeze(1), 2, 2).squeeze(1)
    l += 0.5 * F.l1_loss(d2p, d2g)
    d4p = F.avg_pool2d(d2p.unsqueeze(1), 2, 2).squeeze(1)
    d4g = F.avg_pool2d(d2g.unsqueeze(1), 2, 2).squeeze(1)
    l += 0.25 * F.l1_loss(d4p, d4g)
    return l

def low_high_weight(H, W, frac=0.25, w_low=3.0, device='cpu'):
    wy = int(W * frac / 2); wx = int(H * frac / 2)
    w = torch.ones(1, 1, H, W, device=device)
    w[:, :, H//2-wx:H//2+wx, W//2-wy:W//2+wy] = w_low
    return w

# ==== background mask & loss (bg suppression) ====
@torch.no_grad()
def build_bg_mask(img_mb: torch.Tensor, thr: float = 0.88, dilate: int = 5) -> torch.Tensor:
    """
    img_mb: [B, H, W], SoS magnitude of the input MB (or x_t)
    Returns bg_mask: [B, 1, H, W] with 1 on background, 0 on (dilated) foreground
    """
    assert img_mb.ndim == 3
    B, H, W = img_mb.shape
    # per-sample robust normalization by 99.9 percentile
    vals = img_mb.view(B, -1)
    q = torch.quantile(vals, 0.999, dim=1, keepdim=True) + 1e-6
    img_n = (img_mb / q.view(B, 1, 1)).clamp_(0, 1)
    fg = (img_n > thr).float().unsqueeze(1)  # [B,1,H,W]
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
    bg = build_bg_mask(img_mb.detach(), thr=thr, dilate=dilate)  # [B,1,H,W]
    x = img_pred.unsqueeze(1)  # [B,1,H,W]
    # image energy on background
    L_bg_img = torch.mean(bg * torch.abs(x))
    # simple high-pass: subtract 7x7 average pool
    lp = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    hp = torch.abs(x - lp)
    L_bg_hpf = torch.mean(bg * hp)
    L_bg = L_bg_img + hpf_w * L_bg_hpf
    return L_bg_img, L_bg_hpf, L_bg

# ==== Model ====
class UNetSmall(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=128, base_ch=128, k_big=(3,9), T=64,
                 use_xattn=False, xattn_heads=4, xattn_window=6, xattn_mlp=2.0,
                 use_aux_k23=False):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim, T=T)
        self.two_stream = True
        self.use_xattn = use_xattn
        self.use_aux_k23 = use_aux_k23

        # encoder level 1 (shared)
        self.enc1  = ConvBlock(in_ch, base_ch, time_dim)
        self.down1 = nn.MaxPool2d(2)
        # encoder level 2 (shared -> split dual-stream)
        ch2 = base_ch * 2
        self.enc2  = ConvBlock(base_ch, ch2, time_dim)
        self.down2 = nn.MaxPool2d(2)
        self.split_e2 = nn.Conv2d(ch2, ch2*2, 1, bias=False)
        self.merge_e2 = nn.Conv2d(ch2*2, ch2, 1, bias=False)
        self.cross_e2 = CrossInteractionBlock(ch2, time_dim=time_dim)
        # bottleneck split-cross-merge (dual-stream + optional xattn)
        self.split_mid = nn.Conv2d(ch2, ch2*2, 1, bias=False)
        self.merge_mid = nn.Conv2d(ch2*2, ch2, 1, bias=False)
        self.cross_mid = CrossInteractionBlock(ch2, time_dim=time_dim)
        if self.use_xattn:
            self.xattn = DualStreamXAttnBlock(ch2, heads=xattn_heads, window=xattn_window, mlp_ratio=xattn_mlp)
        else:
            self.xattn = None
        self.mid  = ConvBlock(ch2, ch2, time_dim)
        # ===== Decoder A/B 双流上采样 =====
        self.split_dec0 = nn.Conv2d(ch2, ch2*2, 1, bias=False)
        # stage 1 @ H/2
        self.up1_a  = UpsamplePS(in_ch=ch2, mid_ch=ch2, out_ch=ch2)
        self.up1_b  = UpsamplePS(in_ch=ch2, mid_ch=ch2, out_ch=ch2)
        self.skip2_to_a = nn.Conv2d(ch2, ch2, 1, bias=False)
        self.skip2_to_b = nn.Conv2d(ch2, ch2, 1, bias=False)
        self.fuse1_a= nn.Conv2d(ch2*2, ch2, 1, bias=False)
        self.fuse1_b= nn.Conv2d(ch2*2, ch2, 1, bias=False)
        self.dec1_a = LKRSEFiLM(ch=ch2, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.dec1_b = LKRSEFiLM(ch=ch2, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.cross_dec1 = CrossInteractionBlock(ch2, time_dim=time_dim)
        # stage 2 @ H
        self.up2_a  = UpsamplePS(in_ch=ch2, mid_ch=base_ch, out_ch=base_ch)
        self.up2_b  = UpsamplePS(in_ch=ch2, mid_ch=base_ch, out_ch=base_ch)
        self.skip1_to_a = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.skip1_to_b = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.fuse2_a= nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.fuse2_b= nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.dec2_a = LKRSEFiLM(ch=base_ch, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.dec2_b = LKRSEFiLM(ch=base_ch, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.cross_dec2 = CrossInteractionBlock(base_ch, time_dim=time_dim)
        # merge 两支后输出主头 delta
        self.merge_out = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.out_conv  = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        # 轻量 K23 辅助头（仅训练用）：直接用 B 流末端 c2_b
        if self.use_aux_k23:
            self.k23_head = nn.Sequential(
                nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False), _gn(base_ch), nn.SiLU(),
                nn.Conv2d(base_ch, out_ch, 3, padding=1, bias=True)
            )
        else:
            self.k23_head = None

    def forward(self, x, t, return_feats: bool = False):
        t_emb = self.time_emb(t)
        # enc1
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        # enc2
        e2_shared = self.enc2(d1, t_emb)
        f2 = self.split_e2(e2_shared)
        f2_a, f2_b = torch.chunk(f2, 2, dim=1)
        f2_a, f2_b = self.cross_e2(f2_a, f2_b, t_emb)
        e2_mix = self.merge_e2(torch.cat([f2_a, f2_b], dim=1))
        # bottleneck
        d2 = self.down2(e2_mix)
        fmid = self.split_mid(d2)
        mid_a, mid_b = torch.chunk(fmid, 2, dim=1)
        mid_a, mid_b = self.cross_mid(mid_a, mid_b, t_emb)
        if self.use_xattn:
            mid_a, mid_b = self.xattn(mid_a, mid_b)
        mid_mix = self.merge_mid(torch.cat([mid_a, mid_b], dim=1))
        m  = self.mid(mid_mix, t_emb)
        # ===== Decoder A/B 双流上采样 =====
        dec0 = self.split_dec0(m)
        d0_a, d0_b = torch.chunk(dec0, 2, dim=1)
        # stage 1 (H/2)
        u1_a = self.up1_a(d0_a); u1_b = self.up1_b(d0_b)
        s2_a = self.skip2_to_a(e2_mix); s2_b = self.skip2_to_b(e2_mix)
        c1_a = self.fuse1_a(torch.cat([u1_a, s2_a], dim=1))
        c1_b = self.fuse1_b(torch.cat([u1_b, s2_b], dim=1))
        c1_a = self.dec1_a(c1_a, t_emb); c1_b = self.dec1_b(c1_b, t_emb)
        c1_a, c1_b = self.cross_dec1(c1_a, c1_b, t_emb)
        # stage 2 (H)
        u2_a = self.up2_a(c1_a); u2_b = self.up2_b(c1_b)
        s1_a = self.skip1_to_a(e1);  s1_b = self.skip1_to_b(e1)
        c2_a = self.fuse2_a(torch.cat([u2_a, s1_a], dim=1))
        c2_b = self.fuse2_b(torch.cat([u2_b, s1_b], dim=1))
        c2_a = self.dec2_a(c2_a, t_emb); c2_b = self.dec2_b(c2_b, t_emb)
        c2_a, c2_b = self.cross_dec2(c2_a, c2_b, t_emb)
        # merge & out
        c_final = self.merge_out(torch.cat([c2_a, c2_b], dim=1))
        delta_out = self.out_conv(c_final)
        if return_feats:
            fd = {'mid_a': mid_a, 'mid_b': mid_b, 'dec_a': c2_a, 'dec_b': c2_b}
            if self.use_aux_k23 and (self.k23_head is not None):
                fd['k23_pred'] = self.k23_head(c2_b)
            return delta_out, fd
        return delta_out

# ==== training loop ====

def train():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    Ht, Wt = None, None
    if args.pad_to:
        parts = re.split('[x,]', args.pad_to)
        if len(parts) == 2: Ht, Wt = int(parts[0]), int(parts[1])

    if args.train_dir:
        ds_train = MB3FolderDataset(args.train_dir, T=args.T, alpha_sched=args.alpha_sched,
                                    amp_norm=args.amp_norm, amp_norm_mode=args.amp_norm_mode,
                                    sum23_div_sqrt2=args.sum23_div_sqrt2,
                                    target_hw=(Ht, Wt) if (Ht and Wt) else None,
                                    coils=args.coils, tsample=args.tsample)
    else:
        raise ValueError('--train_dir 不能为空')

    ds_val = None
    if args.val_dir:
        ds_val = MB3FolderDataset(args.val_dir, T=args.T, alpha_sched=args.alpha_sched,
                                  amp_norm=args.amp_norm, amp_norm_mode=args.amp_norm_mode,
                                  sum23_div_sqrt2=args.sum23_div_sqrt2,
                                  target_hw=(Ht, Wt) if (Ht and Wt) else None,
                                  coils=args.coils, tsample='uniform')

    viz_epoch_dir = os.path.join(args.viz_dir, 'epoch_val')
    os.makedirs(viz_epoch_dir, exist_ok=True)

    dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False,
                    persistent_workers=(args.num_workers>0), prefetch_factor=4)

    C = args.coils; in_ch = out_ch = 2*C
    net = UNetSmall(in_ch, out_ch, time_dim=128, base_ch=args.base_ch, k_big=(3,9), T=args.T,
                    use_xattn=args.xattn, xattn_heads=args.xattn_heads, xattn_window=args.xattn_window,
                    xattn_mlp=args.xattn_mlp, use_aux_k23=args.use_aux_k23).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = _amp_scaler('cuda') if device == 'cuda' else None
    ema = EMA(net, decay=args.ema_decay) if args.ema else None

    total_steps = args.epochs * max(1, len(dl)); warmup_steps = int(0.05 * total_steps)
    def lr_schedule(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    best_metric = -1.0

    for ep in range(1, args.epochs + 1):
        net.train(); loss_meter = 0.0
        pbar = tqdm(dl, total=len(dl), desc=f'Epoch {ep}/{args.epochs}', dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            if len(batch) == 5:
                x_t, t_idx, K1_gt, K23_gt, _ = batch
            else:
                x_t, t_idx, K1_gt, K23_gt = batch
            x_t = x_t.to(device, non_blocking=True)
            t_idx = t_idx.to(device, non_blocking=True)
            K1_gt = K1_gt.to(device, non_blocking=True)
            K23_gt = K23_gt.to(device, non_blocking=True)

            amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            with _amp_autocast(enabled=(scaler is not None), dtype=amp_dtype):
                delta_pred, feat = net(x_t, t_idx, return_feats=True)
                K1_pred = x_t - delta_pred
                # 基线损失
                w = low_high_weight(K1_pred.size(-2), K1_pred.size(-1),
                                    frac=args.lowfreq_frac, w_low=args.w_low, device=device)
                l1_k = (w * torch.abs(to_complex(K1_pred - K1_gt))).mean()
                img_pred = k2img_sos_from_2cchw(K1_pred)
                img_gt   = k2img_sos_from_2cchw(K1_gt)
                l1_img   = ms_l1(img_pred, img_gt)
                a_vals = torch.tensor([ds_train.alpha(int(t)) for t in t_idx.detach().cpu().tolist()],
                                      device=device).view(-1, 1, 1, 1)
                x_rec = forward_to_mb(K1_pred, K23_gt, alpha=1.0 * a_vals)
                cyc = torch.mean(torch.abs(to_complex(x_rec - x_t)))
                K23_est = x_t - K1_pred
                diff = torch.abs(to_complex(K23_est) - to_complex(a_vals * K23_gt))
                L_k2_per = diff.view(diff.size(0), -1).mean(dim=1)
                w_t = (4 * a_vals.view(-1) * (1 - a_vals.view(-1)))
                L_k2 = (w_t * L_k2_per).mean()
                L_dec = torch.tensor(0.0, device=device)
                mid_a, mid_b = feat.get('mid_a', None), feat.get('mid_b', None)
                if args.decouple_w > 0 and (mid_a is not None) and (mid_b is not None):
                    K1_c = to_complex(K1_gt)
                    K23_c = to_complex(a_vals * K23_gt)
                    mag1 = torch.sqrt(torch.sum(torch.real(K1_c)**2 + torch.imag(K1_c)**2,
                                                dim=1, keepdim=True) + 1e-8)
                    mag2 = torch.sqrt(torch.sum(torch.real(K23_c)**2 + torch.imag(K23_c)**2,
                                                dim=1, keepdim=True) + 1e-8)
                    denom = mag1 + mag2 + 1e-8
                    w1 = mag1 / denom; w2 = mag2 / denom
                    H_mid, W_mid = mid_a.shape[-2], mid_a.shape[-1]
                    if w1.shape[-2:] != (H_mid, W_mid):
                        w1_mid = F.interpolate(w1, size=(H_mid, W_mid), mode='bilinear', align_corners=False)
                        w2_mid = F.interpolate(w2, size=(H_mid, W_mid), mode='bilinear', align_corners=False)
                    else:
                        w1_mid, w2_mid = w1, w2
                    ea = torch.mean(mid_a**2, dim=1, keepdim=True)
                    eb = torch.mean(mid_b**2, dim=1, keepdim=True)
                    L_dec = (w1_mid * eb + w2_mid * ea).mean()
                # ===== AUX-K23: 三项可选损失 =====
                L_fc = torch.tensor(0.0, device=device)
                L_k23 = torch.tensor(0.0, device=device)
                L_orth = torch.tensor(0.0, device=device)
                if args.use_aux_k23 and ('k23_pred' in feat):
                    k23_pred = feat['k23_pred']
                    # 1) 前向一致性：K1_pred + a_t * k23_pred ≈ x_t
                    x_hat = forward_to_mb(K1_pred, k23_pred, alpha=1.0 * a_vals)
                    L_fc = torch.mean(torch.abs(to_complex(x_hat - x_t)))
                    # 2) K23 监督（可选）
                    if args.k23_w > 0:
                        L_k23 = torch.mean(torch.abs(to_complex(k23_pred - K23_gt)))
                    # 3) 正交惩罚：cosine similarity on RI-stacked vectors
                    v1 = K1_pred.view(K1_pred.size(0), -1)
                    v2 = k23_pred.view(k23_pred.size(0), -1)
                    num = torch.sum(v1 * v2, dim=1)
                    den = torch.sqrt(torch.sum(v1*v1, dim=1) + 1e-12) * torch.sqrt(torch.sum(v2*v2, dim=1) + 1e-12)
                    cos = num / (den + 1e-12)
                    L_orth = torch.mean(torch.abs(cos))
                # === Background suppression loss ===
                L_bg = torch.tensor(0.0, device=device)
                if args.bg_w > 0:
                    # use x_t (current MB mixture) to build a stable mask
                    img_mb = k2img_sos_from_2cchw(x_t)
                    L_bg_img, L_bg_hpf, L_bg = bg_losses(img_pred, img_mb, thr=args.bg_thr, dilate=args.bg_dilate, hpf_w=args.bg_hpf)
                loss = (l1_k + args.img_w * l1_img + args.cycle_w * cyc
                        + args.k2_w * L_k2 + args.decouple_w * L_dec
                        + args.fc_w * L_fc + args.k23_w * L_k23 + args.orth_w * L_orth + args.bg_w * L_bg)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
            if ema: ema.update(net)
            scheduler.step()
            loss_meter += loss.item()
            if step % 10 == 0 or step == 1:
                pbar.set_postfix(loss=loss_meter/step)

        # save ckpt per epoch
        ckpt_path = os.path.join(args.save_dir, f'mb3k1_epoch{ep:03d}.pt')
        torch.save({'epoch': ep, 'model': net.state_dict(),
                    'opt': opt.state_dict(), 'args': vars(args)}, ckpt_path)

        # quick val
        if ds_val is not None:
            backup_sd = None
            if ema:
                backup_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
                ema.copy_to(net)
            net.eval(); psnr_list = []; best_triplet = None; best_psnr = -1.0
            with torch.no_grad():
                N = min(args.val_samples, len(ds_val))
                import random as _rnd
                idxs = _rnd.sample(range(len(ds_val)), N)
                for i in idxs:
                    sample = ds_val[i]
                    if len(sample) == 5:
                        _, _, K1_gt_val, K23_val, KMB_val = sample
                    else:
                        _, _, K1_gt_val, K23_val = sample
                        KMB_val = None
                    K1_gt_val = K1_gt_val.to(device)[None]
                    K23_val = K23_val.to(device)[None]
                    if (KMB_val is not None) and args.val_use_file_kmb:
                        x_mb = KMB_val.to(device)[None] if KMB_val.ndim == 3 else KMB_val.to(device)
                    else:
                        x_mb = forward_to_mb(K1_gt_val, K23_val, alpha=1.0)
                    K1_out = infer_mb_to_k1(net, x_mb, T=args.T, alpha_sched=args.alpha_sched)
                    img_pred = k2img_sos_from_2cchw(K1_out[0])
                    img_gt   = k2img_sos_from_2cchw(K1_gt_val[0])
                    img_mb   = k2img_sos_from_2cchw(x_mb[0])
                    pred_np = img_pred.cpu().numpy().astype(np.float32)
                    gt_np   = img_gt.cpu().numpy().astype(np.float32)
                    mse = float(np.mean((pred_np - gt_np)**2))
                    max_signal = float(gt_np.max())
                    psnr_val = 20 * np.log10(max_signal / (np.sqrt(mse) + 1e-8))
                    psnr_list.append(psnr_val)
                    if psnr_val > best_psnr:
                        best_psnr = float(psnr_val)
                        best_triplet = (_to_gray01(img_gt), _to_gray01(img_mb), _to_gray01(img_pred))
            avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
            max_psnr = float(np.max(psnr_list)) if psnr_list else 0.0
            print(f'Epoch {ep}: Val subset (N={N}) mean PSNR = {avg_psnr:.2f} dB')
            print(f'Epoch {ep}: Val subset (N={N}) max  PSNR = {max_psnr:.2f} dB')
            if best_triplet is not None:
                titles = ['GT K1', 'Input MB', f'Pred K1 (PSNR={best_psnr:.2f} dB)']
                out_path = os.path.join(args.viz_dir, 'epoch_val', f'epoch{ep:03d}_random_MB.png')
                save_grid(list(best_triplet), titles, cols=3, out_path=out_path)
            metric = avg_psnr
            if metric > best_metric:
                best_metric = metric
                best_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({'epoch': ep, 'model': net.state_dict(),
                            'psnr_mean': avg_psnr, 'psnr_max': max_psnr}, best_path)
            if ema and backup_sd is not None:
                net.load_state_dict(backup_sd)

    print('训练完成！')

# ==== args ====

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, default='')
    p.add_argument('--val_dir',   type=str, default='')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--alpha_sched', type=str, default='cos', choices=['lin','cos'])
    p.add_argument('--lowfreq_frac', type=float, default=0.25)
    p.add_argument('--w_low', type=float, default=3.0)
    p.add_argument('--cycle_w', type=float, default=0.1)
    p.add_argument('--img_w', type=float, default=0.1)
    p.add_argument('--k2_w', type=float, default=0.25)
    p.add_argument('--decouple_w', type=float, default=0.05)
    p.add_argument('--amp_norm', action='store_true')
    p.add_argument('--amp_norm_mode', type=str, default='kmb', choices=['k1','kmb'])
    p.add_argument('--sum23_div_sqrt2', action='store_true')
    p.add_argument('--pad_to', type=str, default='')
    p.add_argument('--coils', type=int, default=12)
    p.add_argument('--tsample', type=str, default='uniform', choices=['uniform','mid'])
    p.add_argument('--base_ch', type=int, default=128)
    p.add_argument('--ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--val_samples', type=int, default=200)
    p.add_argument('--val_use_file_kmb', action='store_true')
    p.add_argument('--viz_dir', type=str, default='viz')
    p.add_argument('--save_dir', type=str, default='runs_mb3k1')
    p.add_argument('--xattn', action='store_true')
    p.add_argument('--xattn_heads', type=int, default=4)
    p.add_argument('--xattn_window', type=int, default=6)
    p.add_argument('--xattn_mlp', type=float, default=2.0)
    # === AUX-K23 flags ===
    p.add_argument('--use_aux_k23', action='store_true', help='启用轻量 K23 辅助头（训练期有效）')
    p.add_argument('--fc_w', type=float, default=0.0, help='前向一致性损失权重（K1+a·K23≈x_t）')
    p.add_argument('--k23_w', type=float, default=0.0, help='K23 监督损失权重（需有 K2+K3 GT）')
    p.add_argument('--orth_w', type=float, default=0.0, help='K1 与 K23 正交惩罚权重')
    # === Background suppression flags ===
    p.add_argument('--bg_w', type=float, default=0.0, help='背景抑制损失总权重')
    p.add_argument('--bg_thr', type=float, default=0.88, help='前景阈值（基于99.9分位归一化）')
    p.add_argument('--bg_dilate', type=int, default=5, help='前景掩膜膨胀像素')
    p.add_argument('--bg_hpf', type=float, default=0.5, help='背景高频惩罚的相对权重')
    return p.parse_args()


if __name__ == '__main__':
    train()
