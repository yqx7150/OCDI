from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(ch, groups=8):
    g = groups if ch % groups == 0 and ch >= groups else 1
    return nn.GroupNorm(g, ch)


class SE(nn.Module):

    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(ch // r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(ch, hidden, 1, bias=True), nn.SiLU(), nn.Conv2d(hidden, ch, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.avg(x))


class TimeEmbedding(nn.Module):

    def __init__(self, dim=128, T=64):
        super().__init__()
        self.emb = nn.Embedding(T + 1, dim)

    def forward(self, t):
        return self.emb(t)


class FiLM(nn.Module):

    def __init__(self, time_dim, ch):
        super().__init__()
        self.mlp = nn.Linear(time_dim, ch * 2)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, t_emb):
        g, b = torch.chunk(self.mlp(t_emb)[:, :, None, None], 2, dim=1)
        return x * (1 + g) + b


class LKRSEFiLM(nn.Module):

    def __init__(self, ch, time_dim, k=(3, 9), res_scale=0.1):
        super().__init__()
        if isinstance(k, tuple):
            padding = (k[0] // 2, k[1] // 2)
            self.dw = nn.Conv2d(ch, ch, k, padding=padding, groups=ch, bias=False)
        else:
            self.dw = nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.gn1 = _gn(ch)
        self.gn2 = _gn(ch)
        self.act = nn.SiLU()
        self.film = FiLM(time_dim, ch)
        self.se = SE(ch)
        self.res_scale = res_scale

    def forward(self, x, t_emb):
        idt = x
        y = self.dw(x)
        y = self.gn1(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.gn2(y)
        y = self.film(y, t_emb)
        y = self.act(y)
        y = self.se(y)
        return idt + self.res_scale * y


class CrossInteractionBlock(nn.Module):

    def __init__(self, ch, time_dim=None):
        super().__init__()
        self.conv_mix = nn.Conv2d(ch * 2, ch * 2, 3, padding=1, bias=False)
        self.norm = _gn(ch * 2)
        self.act = nn.SiLU()
        self.film = FiLM(time_dim, ch * 2) if time_dim is not None else None
        self.proj_a = nn.Conv2d(ch * 2, ch, 1, bias=False)
        self.proj_b = nn.Conv2d(ch * 2, ch, 1, bias=False)

    def forward(self, a, b, t_emb=None):
        h = torch.cat([a, b], dim=1)
        h = self.conv_mix(h)
        h = self.norm(h)
        if self.film is not None and t_emb is not None:
            h = self.film(h, t_emb)
        h = self.act(h)
        g_a = self.proj_a(h)
        g_b = self.proj_b(h)
        return (a + g_a, b + g_b)


class UpsamplePS(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, mid_ch * 4, 1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.bn1 = _gn(mid_ch)
        self.bn2 = _gn(out_ch)
        self.conv = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.act = nn.SiLU()
        with torch.no_grad():
            out_c, in_c, k1, k2 = self.pre.weight.shape
            subc = out_c // 4
            weight = torch.zeros(out_c, in_c, k1, k2)
            nn.init.kaiming_normal_(weight[:subc])
            weight = weight.view(subc, 4, in_c, k1, k2)
            weight = weight.permute(1, 0, 2, 3, 4).contiguous().view(out_c, in_c, k1, k2)
            self.pre.weight.copy_(weight)

    def forward(self, x):
        x = self.ps(self.pre(x))
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(time_dim, out_ch)

    def forward(self, x, t_emb):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, t_emb)
        return self.act(x)


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
    H = H0 + pad_h
    W = W0 + pad_w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    x = x.view(B, C, H // Wh, Wh, W // Ww, Ww)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, Wh * Ww, C)
    meta = (B, C, H0, W0, H, W, Wh, Ww)
    return (windows, meta)


def window_reverse(windows, meta):
    B, C, H0, W0, H, W, Wh, Ww = meta
    nH, nW = (H // Wh, W // Ww)
    x = windows.view(B, nH, nW, Wh, Ww, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    if H != H0 or W != W0:
        x = x[:, :, :H0, :W0]
    return x


def _make_divisible(x, m):
    if x % m == 0:
        return x
    y = x - x % m
    return max(y, m)


class WindowMHA(nn.Module):

    def __init__(self, ch, num_heads=4, window_size=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window = _to_2tuple(window_size)
        embed_dim = _make_divisible(ch, num_heads)
        self.in_proj = nn.Conv2d(ch, embed_dim, 1, bias=True)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Conv2d(embed_dim, ch, 1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = (embed_dim // num_heads) ** (-0.5)

    def forward(self, x):
        B, C, H0, W0 = x.shape
        x_embed = self.in_proj(x)
        windows, meta = window_partition(x_embed, self.window)
        qkv = self.qkv(windows)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = q.shape[-1] // self.num_heads
        q = q.view(q.shape[0], q.shape[1], self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, head_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(windows.shape[0], windows.shape[1], -1)
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
        self.proj_q = nn.Conv2d(ch_q, embed_q, 1, bias=True)
        self.proj_kv = nn.Conv2d(ch_kv, embed_kv, 1, bias=True)
        self.q_proj = nn.Linear(embed_q, embed_q, bias=True)
        self.kv_proj = nn.Linear(embed_kv, embed_kv * 2, bias=True)
        self.proj = nn.Linear(embed_q, embed_q, bias=True)
        self.out = nn.Conv2d(embed_q, ch_q, 1, bias=True)
        self.scale = (embed_q // num_heads) ** (-0.5)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Qa, Kb):
        B, Ca, H0, W0 = Qa.shape
        _, Cb, Hb, Wb = Kb.shape
        assert H0 == Hb and W0 == Wb, 'CrossAttn expects same spatial size for A/B'
        Qa_e = self.proj_q(Qa)
        Kb_e = self.proj_kv(Kb)
        Qwin, meta = window_partition(Qa_e, self.window)
        KVwin, _ = window_partition(Kb_e, self.window)
        q = self.q_proj(Qwin)
        kv = self.kv_proj(KVwin)
        k, v = kv.chunk(2, dim=-1)
        head_dim = q.shape[-1] // self.num_heads
        q = q.view(q.shape[0], q.shape[1], self.num_heads, head_dim).permute(0, 2, 1, 3)
        head_dim_k = k.shape[-1] // self.num_heads
        k = k.view(k.shape[0], k.shape[1], self.num_heads, head_dim_k).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, head_dim_k).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(Qwin.shape[0], Qwin.shape[1], -1)
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
        self.norm_a1 = _gn(ch)
        self.norm_b1 = _gn(ch)
        self.norm_a2 = _gn(ch)
        self.norm_b2 = _gn(ch)
        self.mlp_a = MLP2d(ch, mlp_ratio=mlp_ratio)
        self.mlp_b = MLP2d(ch, mlp_ratio=mlp_ratio)

    def forward(self, a, b):
        a1 = a + self.sa_a(self.norm_a1(a))
        b1 = b + self.sa_b(self.norm_b1(b))
        a2 = a1 + self.ca_ab(self.norm_a2(a1), self.norm_b2(b1))
        b2 = b1 + self.ca_ba(self.norm_b2(b1), self.norm_a2(a1))
        return (a2 + self.mlp_a(a2), b2 + self.mlp_b(b2))
