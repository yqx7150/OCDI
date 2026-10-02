from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocdi.data.mb3 import MB3FolderDataset
from ocdi.engine.ema import EMA
from ocdi.engine.inference import infer_mb_to_k1
from ocdi.engine.losses import bg_losses, low_high_weight, ms_l1
from ocdi.net import OCDINet
from ocdi.ops.complex import forward_to_mb, k2img_sos_from_2cchw, to_complex
from ocdi.utils.amp import _amp_autocast, _amp_scaler
from ocdi.utils.visualization import save_grid, tensor_to_gray01

def train():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    Ht, Wt = (None, None)
    if args.pad_to:
        parts = re.split('[x,]', args.pad_to)
        if len(parts) == 2:
            Ht, Wt = (int(parts[0]), int(parts[1]))
    if args.train_dir:
        ds_train = MB3FolderDataset(args.train_dir, T=args.T, alpha_sched=args.alpha_sched, amp_norm=args.amp_norm, amp_norm_mode=args.amp_norm_mode, sum23_div_sqrt2=args.sum23_div_sqrt2, target_hw=(Ht, Wt) if Ht and Wt else None, coils=args.coils, tsample=args.tsample)
    else:
        raise ValueError('--train_dir is required')
    ds_val = None
    if args.val_dir:
        ds_val = MB3FolderDataset(args.val_dir, T=args.T, alpha_sched=args.alpha_sched, amp_norm=args.amp_norm, amp_norm_mode=args.amp_norm_mode, sum23_div_sqrt2=args.sum23_div_sqrt2, target_hw=(Ht, Wt) if Ht and Wt else None, coils=args.coils, tsample='uniform')
    viz_epoch_dir = os.path.join(args.viz_dir, 'epoch_val')
    os.makedirs(viz_epoch_dir, exist_ok=True)
    dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0, prefetch_factor=4)
    C = args.coils
    in_ch = out_ch = 2 * C
    net = OCDINet(in_ch, out_ch, time_dim=128, base_ch=args.base_ch, k_big=(3, 9), T=args.T, use_xattn=args.xattn, xattn_heads=args.xattn_heads, xattn_window=args.xattn_window, xattn_mlp=args.xattn_mlp, use_aux_k23=args.use_aux_k23).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = _amp_scaler('cuda') if device == 'cuda' else None
    ema = EMA(net, decay=args.ema_decay) if args.ema else None
    total_steps = args.epochs * max(1, len(dl))
    warmup_steps = int(0.05 * total_steps)

    def lr_schedule(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)
    best_metric = -1.0
    for ep in range(1, args.epochs + 1):
        net.train()
        loss_meter = 0.0
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
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with _amp_autocast(enabled=scaler is not None, dtype=amp_dtype):
                delta_pred, feat = net(x_t, t_idx, return_feats=True)
                K1_pred = x_t - delta_pred
                w = low_high_weight(K1_pred.size(-2), K1_pred.size(-1), frac=args.lowfreq_frac, w_low=args.w_low, device=device)
                l1_k = (w * torch.abs(to_complex(K1_pred - K1_gt))).mean()
                img_pred = k2img_sos_from_2cchw(K1_pred)
                img_gt = k2img_sos_from_2cchw(K1_gt)
                l1_img = ms_l1(img_pred, img_gt)
                a_vals = torch.tensor([ds_train.alpha(int(t)) for t in t_idx.detach().cpu().tolist()], device=device).view(-1, 1, 1, 1)
                x_rec = forward_to_mb(K1_pred, K23_gt, alpha=1.0 * a_vals)
                cyc = torch.mean(torch.abs(to_complex(x_rec - x_t)))
                K23_est = x_t - K1_pred
                diff = torch.abs(to_complex(K23_est) - to_complex(a_vals * K23_gt))
                L_k2_per = diff.view(diff.size(0), -1).mean(dim=1)
                w_t = 4 * a_vals.view(-1) * (1 - a_vals.view(-1))
                L_k2 = (w_t * L_k2_per).mean()
                L_dec = torch.tensor(0.0, device=device)
                mid_a, mid_b = (feat.get('mid_a', None), feat.get('mid_b', None))
                if args.decouple_w > 0 and mid_a is not None and (mid_b is not None):
                    K1_c = to_complex(K1_gt)
                    K23_c = to_complex(a_vals * K23_gt)
                    mag1 = torch.sqrt(torch.sum(torch.real(K1_c) ** 2 + torch.imag(K1_c) ** 2, dim=1, keepdim=True) + 1e-08)
                    mag2 = torch.sqrt(torch.sum(torch.real(K23_c) ** 2 + torch.imag(K23_c) ** 2, dim=1, keepdim=True) + 1e-08)
                    denom = mag1 + mag2 + 1e-08
                    w1 = mag1 / denom
                    w2 = mag2 / denom
                    H_mid, W_mid = (mid_a.shape[-2], mid_a.shape[-1])
                    if w1.shape[-2:] != (H_mid, W_mid):
                        w1_mid = F.interpolate(w1, size=(H_mid, W_mid), mode='bilinear', align_corners=False)
                        w2_mid = F.interpolate(w2, size=(H_mid, W_mid), mode='bilinear', align_corners=False)
                    else:
                        w1_mid, w2_mid = (w1, w2)
                    ea = torch.mean(mid_a ** 2, dim=1, keepdim=True)
                    eb = torch.mean(mid_b ** 2, dim=1, keepdim=True)
                    L_dec = (w1_mid * eb + w2_mid * ea).mean()
                L_fc = torch.tensor(0.0, device=device)
                L_k23 = torch.tensor(0.0, device=device)
                L_orth = torch.tensor(0.0, device=device)
                if args.use_aux_k23 and 'k23_pred' in feat:
                    k23_pred = feat['k23_pred']
                    x_hat = forward_to_mb(K1_pred, k23_pred, alpha=1.0 * a_vals)
                    L_fc = torch.mean(torch.abs(to_complex(x_hat - x_t)))
                    if args.k23_w > 0:
                        L_k23 = torch.mean(torch.abs(to_complex(k23_pred - K23_gt)))
                    v1 = K1_pred.view(K1_pred.size(0), -1)
                    v2 = k23_pred.view(k23_pred.size(0), -1)
                    num = torch.sum(v1 * v2, dim=1)
                    den = torch.sqrt(torch.sum(v1 * v1, dim=1) + 1e-12) * torch.sqrt(torch.sum(v2 * v2, dim=1) + 1e-12)
                    cos = num / (den + 1e-12)
                    L_orth = torch.mean(torch.abs(cos))
                L_bg = torch.tensor(0.0, device=device)
                if args.bg_w > 0:
                    img_mb = k2img_sos_from_2cchw(x_t)
                    L_bg_img, L_bg_hpf, L_bg = bg_losses(img_pred, img_mb, thr=args.bg_thr, dilate=args.bg_dilate, hpf_w=args.bg_hpf)
                loss = l1_k + args.img_w * l1_img + args.cycle_w * cyc + args.k2_w * L_k2 + args.decouple_w * L_dec + args.fc_w * L_fc + args.k23_w * L_k23 + args.orth_w * L_orth + args.bg_w * L_bg
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
            if ema:
                ema.update(net)
            scheduler.step()
            loss_meter += loss.item()
            if step % 10 == 0 or step == 1:
                pbar.set_postfix(loss=loss_meter / step)
        ckpt_path = os.path.join(args.save_dir, f'mb3k1_epoch{ep:03d}.pt')
        torch.save({'epoch': ep, 'model': net.state_dict(), 'opt': opt.state_dict(), 'args': vars(args)}, ckpt_path)
        if ds_val is not None:
            backup_sd = None
            if ema:
                backup_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
                ema.copy_to(net)
            net.eval()
            psnr_list = []
            best_triplet = None
            best_psnr = -1.0
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
                    if KMB_val is not None and args.val_use_file_kmb:
                        x_mb = KMB_val.to(device)[None] if KMB_val.ndim == 3 else KMB_val.to(device)
                    else:
                        x_mb = forward_to_mb(K1_gt_val, K23_val, alpha=1.0)
                    K1_out = infer_mb_to_k1(net, x_mb, T=args.T, alpha_sched=args.alpha_sched)
                    img_pred = k2img_sos_from_2cchw(K1_out[0])
                    img_gt = k2img_sos_from_2cchw(K1_gt_val[0])
                    img_mb = k2img_sos_from_2cchw(x_mb[0])
                    pred_np = img_pred.cpu().numpy().astype(np.float32)
                    gt_np = img_gt.cpu().numpy().astype(np.float32)
                    mse = float(np.mean((pred_np - gt_np) ** 2))
                    max_signal = float(gt_np.max())
                    psnr_val = 20 * np.log10(max_signal / (np.sqrt(mse) + 1e-08))
                    psnr_list.append(psnr_val)
                    if psnr_val > best_psnr:
                        best_psnr = float(psnr_val)
                        best_triplet = (tensor_to_gray01(img_gt), tensor_to_gray01(img_mb), tensor_to_gray01(img_pred))
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
                torch.save({'epoch': ep, 'model': net.state_dict(), 'psnr_mean': avg_psnr, 'psnr_max': max_psnr}, best_path)
            if ema and backup_sd is not None:
                net.load_state_dict(backup_sd)
    print('Training complete.')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, default='')
    p.add_argument('--val_dir', type=str, default='')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--alpha_sched', type=str, default='cos', choices=['lin', 'cos'])
    p.add_argument('--lowfreq_frac', type=float, default=0.25)
    p.add_argument('--w_low', type=float, default=3.0)
    p.add_argument('--cycle_w', type=float, default=0.1)
    p.add_argument('--img_w', type=float, default=0.1)
    p.add_argument('--k2_w', type=float, default=0.25)
    p.add_argument('--decouple_w', type=float, default=0.05)
    p.add_argument('--amp_norm', action='store_true')
    p.add_argument('--amp_norm_mode', type=str, default='kmb', choices=['k1', 'kmb'])
    p.add_argument('--sum23_div_sqrt2', action='store_true')
    p.add_argument('--pad_to', type=str, default='')
    p.add_argument('--coils', type=int, default=12)
    p.add_argument('--tsample', type=str, default='uniform', choices=['uniform', 'mid'])
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
    p.add_argument('--use_aux_k23', action='store_true', help='Enable the auxiliary K23 head during training.')
    p.add_argument('--fc_w', type=float, default=0.0, help='Weight for forward-consistency loss.')
    p.add_argument('--k23_w', type=float, default=0.0, help='Weight for K23 supervision loss.')
    p.add_argument('--orth_w', type=float, default=0.0, help='Weight for K1/K23 orthogonality regularization.')
    p.add_argument('--bg_w', type=float, default=0.0, help='Weight for background suppression loss.')
    p.add_argument('--bg_thr', type=float, default=0.88, help='Foreground threshold after 99.9-percentile normalization.')
    p.add_argument('--bg_dilate', type=int, default=5, help='Foreground mask dilation in pixels.')
    p.add_argument('--bg_hpf', type=float, default=0.5, help='Relative weight of the background high-frequency penalty.')
    return p.parse_args()

if __name__ == '__main__':
    train()
