from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    ConvBlock,
    CrossInteractionBlock,
    DualStreamXAttnBlock,
    LKRSEFiLM,
    TimeEmbedding,
    UpsamplePS,
    _gn,
)

class OCDINet(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim=128, base_ch=128, k_big=(3, 9), T=64, use_xattn=False, xattn_heads=4, xattn_window=6, xattn_mlp=2.0, use_aux_k23=False):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim, T=T)
        self.two_stream = True
        self.use_xattn = use_xattn
        self.use_aux_k23 = use_aux_k23
        self.enc1 = ConvBlock(in_ch, base_ch, time_dim)
        self.down1 = nn.MaxPool2d(2)
        ch2 = base_ch * 2
        self.enc2 = ConvBlock(base_ch, ch2, time_dim)
        self.down2 = nn.MaxPool2d(2)
        self.split_e2 = nn.Conv2d(ch2, ch2 * 2, 1, bias=False)
        self.merge_e2 = nn.Conv2d(ch2 * 2, ch2, 1, bias=False)
        self.cross_e2 = CrossInteractionBlock(ch2, time_dim=time_dim)
        self.split_mid = nn.Conv2d(ch2, ch2 * 2, 1, bias=False)
        self.merge_mid = nn.Conv2d(ch2 * 2, ch2, 1, bias=False)
        self.cross_mid = CrossInteractionBlock(ch2, time_dim=time_dim)
        if self.use_xattn:
            self.xattn = DualStreamXAttnBlock(ch2, heads=xattn_heads, window=xattn_window, mlp_ratio=xattn_mlp)
        else:
            self.xattn = None
        self.mid = ConvBlock(ch2, ch2, time_dim)
        self.split_dec0 = nn.Conv2d(ch2, ch2 * 2, 1, bias=False)
        self.up1_a = UpsamplePS(in_ch=ch2, mid_ch=ch2, out_ch=ch2)
        self.up1_b = UpsamplePS(in_ch=ch2, mid_ch=ch2, out_ch=ch2)
        self.skip2_to_a = nn.Conv2d(ch2, ch2, 1, bias=False)
        self.skip2_to_b = nn.Conv2d(ch2, ch2, 1, bias=False)
        self.fuse1_a = nn.Conv2d(ch2 * 2, ch2, 1, bias=False)
        self.fuse1_b = nn.Conv2d(ch2 * 2, ch2, 1, bias=False)
        self.dec1_a = LKRSEFiLM(ch=ch2, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.dec1_b = LKRSEFiLM(ch=ch2, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.cross_dec1 = CrossInteractionBlock(ch2, time_dim=time_dim)
        self.up2_a = UpsamplePS(in_ch=ch2, mid_ch=base_ch, out_ch=base_ch)
        self.up2_b = UpsamplePS(in_ch=ch2, mid_ch=base_ch, out_ch=base_ch)
        self.skip1_to_a = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.skip1_to_b = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.fuse2_a = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.fuse2_b = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.dec2_a = LKRSEFiLM(ch=base_ch, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.dec2_b = LKRSEFiLM(ch=base_ch, time_dim=time_dim, k=k_big, res_scale=0.1)
        self.cross_dec2 = CrossInteractionBlock(base_ch, time_dim=time_dim)
        self.merge_out = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        if self.use_aux_k23:
            self.k23_head = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False), _gn(base_ch), nn.SiLU(), nn.Conv2d(base_ch, out_ch, 3, padding=1, bias=True))
        else:
            self.k23_head = None

    def forward(self, x, t, return_feats: bool=False):
        t_emb = self.time_emb(t)
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        e2_shared = self.enc2(d1, t_emb)
        f2 = self.split_e2(e2_shared)
        f2_a, f2_b = torch.chunk(f2, 2, dim=1)
        f2_a, f2_b = self.cross_e2(f2_a, f2_b, t_emb)
        e2_mix = self.merge_e2(torch.cat([f2_a, f2_b], dim=1))
        d2 = self.down2(e2_mix)
        fmid = self.split_mid(d2)
        mid_a, mid_b = torch.chunk(fmid, 2, dim=1)
        mid_a, mid_b = self.cross_mid(mid_a, mid_b, t_emb)
        if self.use_xattn:
            mid_a, mid_b = self.xattn(mid_a, mid_b)
        mid_mix = self.merge_mid(torch.cat([mid_a, mid_b], dim=1))
        m = self.mid(mid_mix, t_emb)
        dec0 = self.split_dec0(m)
        d0_a, d0_b = torch.chunk(dec0, 2, dim=1)
        u1_a = self.up1_a(d0_a)
        u1_b = self.up1_b(d0_b)
        s2_a = self.skip2_to_a(e2_mix)
        s2_b = self.skip2_to_b(e2_mix)
        c1_a = self.fuse1_a(torch.cat([u1_a, s2_a], dim=1))
        c1_b = self.fuse1_b(torch.cat([u1_b, s2_b], dim=1))
        c1_a = self.dec1_a(c1_a, t_emb)
        c1_b = self.dec1_b(c1_b, t_emb)
        c1_a, c1_b = self.cross_dec1(c1_a, c1_b, t_emb)
        u2_a = self.up2_a(c1_a)
        u2_b = self.up2_b(c1_b)
        s1_a = self.skip1_to_a(e1)
        s1_b = self.skip1_to_b(e1)
        c2_a = self.fuse2_a(torch.cat([u2_a, s1_a], dim=1))
        c2_b = self.fuse2_b(torch.cat([u2_b, s1_b], dim=1))
        c2_a = self.dec2_a(c2_a, t_emb)
        c2_b = self.dec2_b(c2_b, t_emb)
        c2_a, c2_b = self.cross_dec2(c2_a, c2_b, t_emb)
        c_final = self.merge_out(torch.cat([c2_a, c2_b], dim=1))
        delta_out = self.out_conv(c_final)
        if return_feats:
            fd = {'mid_a': mid_a, 'mid_b': mid_b, 'dec_a': c2_a, 'dec_b': c2_b}
            if self.use_aux_k23 and self.k23_head is not None:
                fd['k23_pred'] = self.k23_head(c2_b)
            return (delta_out, fd)
        return delta_out

UNetSmall = OCDINet
