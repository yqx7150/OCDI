
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from ocdi.data.packed import load_packed_mb3_mat
from ocdi.engine.inference import infer_mb_to_k1
from ocdi.net import OCDINet
from ocdi.ops.caipi import apply_fov_shift_2cchw
from ocdi.ops.complex import quantile_scale_kmb, to_2cchw
from ocdi.utils.metrics import k2img_sos_from_2cchw_matlab, roger_psnr_ssim_matlab_equiv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OCDI evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--slice_ref_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--alpha_sched", type=str, default="cos", choices=["lin", "cos"])
    p.add_argument("--coils", type=int, default=16)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--xattn", action="store_true")
    p.add_argument("--no_xattn", action="store_true")
    p.add_argument("--xattn_heads", type=int, default=4)
    p.add_argument("--xattn_window", type=int, default=6)
    p.add_argument("--xattn_mlp", type=float, default=2.0)
    p.add_argument("--use_aux_k23", action="store_true")
    p.add_argument("--no_aux_k23", action="store_true")
    p.add_argument("--amp_norm", action="store_true")
    p.add_argument("--amp_norm_mode", type=str, default="kmb", choices=["kmb"])
    p.add_argument("--targets", type=str, default="k1,k2,k3")
    p.add_argument("--shift_axis", type=str, default="x", choices=["x", "y"])
    p.add_argument(
        "--shift_fracs",
        type=str,
        default="0,-0.3333333333333333,0.3333333333333333",
    )
    p.add_argument("--ref_lines", "--lf_lines", dest="ref_lines", type=int, default=32)
    p.add_argument("--kernel", type=str, default="5,5")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    p.add_argument("--no_col_whiten", action="store_true")
    p.add_argument("--ref_strength", "--slf_w", dest="ref_strength", type=float, default=0.12)
    p.add_argument("--ref_power", "--slf_pow", dest="ref_power", type=float, default=2.5)
    p.add_argument("--no_guide_delta", action="store_true")
    p.add_argument("--ref_late_step", "--slf_late_step_enable", dest="ref_late_step", action="store_true")
    p.add_argument("--ref_late_start", "--slf_late_start", dest="ref_late_start", type=int, default=12)
    p.add_argument("--ref_beta_max", "--slf_beta_max", dest="ref_beta_max", type=float, default=0.10)
    p.add_argument("--ref_beta_pow", "--slf_beta_pow", dest="ref_beta_pow", type=float, default=2.0)
    p.add_argument("--ref_rest_mix", "--slf_rest_mix", dest="ref_rest_mix", type=float, default=1.0)
    p.add_argument("--ref_fuse_enable", "--slf_fuse_enable", dest="ref_fuse_enable", action="store_true")
    p.add_argument("--ref_fuse_lines", "--slf_fuse_lines", dest="ref_fuse_lines", type=int, default=0)
    p.add_argument("--ref_fuse_shape", "--slf_fuse_shape", dest="ref_fuse_shape", type=str, default="ky", choices=["ky", "radial"])
    p.add_argument("--ref_fuse_mode", "--slf_fuse_mode", dest="ref_fuse_mode", type=str, default="gaussian", choices=["gaussian", "sigmoid", "cosine"])
    p.add_argument("--ref_fuse_transition", "--slf_fuse_transition", dest="ref_fuse_transition", type=float, default=0.0)
    p.add_argument("--ref_min_w", "--slf_min_w", dest="ref_min_w", type=float, default=0.0)
    p.add_argument("--ref_max_w", "--slf_max_w", dest="ref_max_w", type=float, default=1.0)
    p.add_argument("--no_ref_force_center", "--no_slf_force_center", dest="no_ref_force_center", action="store_true")
    p.add_argument("--ref_no_final_fuse", "--slf_no_final_fuse", dest="ref_no_final_fuse", action="store_true")
    p.add_argument("--metric_norm", type=str, default="max", choices=["max", "p99"])
    p.add_argument("--no_match_gain", action="store_true")
    p.add_argument("--no_clip", action="store_true")
    return p


def _load_reference(path: str, coils: int):
    r1, r2, r3, _ = load_packed_mb3_mat(path, coils)
    return np.stack([r1, r2, r3], axis=0).astype(np.complex64)


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    targets = [item.strip().lower() for item in args.targets.split(",") if item.strip()]
    targets = targets or ["k1", "k2", "k3"]
    target_to_idx = {"k1": 0, "k2": 1, "k3": 2}
    invalid_targets = [item for item in targets if item not in target_to_idx]
    if invalid_targets:
        raise ValueError(f"Invalid targets: {invalid_targets}")

    shift_fracs = tuple(float(item.strip()) for item in args.shift_fracs.split(","))
    if len(shift_fracs) != 3:
        raise ValueError("--shift_fracs must contain three values.")
    recenter = {f"k{i + 1}": -shift_fracs[i] for i in range(3)}
    kernel_h, kernel_w = [int(item.strip()) for item in args.kernel.split(",")]

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    coils = int(checkpoint_args.get("coils", args.coils))
    steps = int(checkpoint_args.get("T", args.T))
    alpha_sched = str(checkpoint_args.get("alpha_sched", args.alpha_sched))
    base_ch = int(checkpoint_args.get("base_ch", args.base_ch))
    use_xattn = bool(checkpoint_args.get("xattn", False))
    xattn_heads = int(checkpoint_args.get("xattn_heads", args.xattn_heads))
    xattn_window = int(checkpoint_args.get("xattn_window", args.xattn_window))
    xattn_mlp = float(checkpoint_args.get("xattn_mlp", args.xattn_mlp))
    use_aux_k23 = bool(checkpoint_args.get("use_aux_k23", False))
    if args.xattn:
        use_xattn = True
    if args.no_xattn:
        use_xattn = False
    if args.use_aux_k23:
        use_aux_k23 = True
    if args.no_aux_k23:
        use_aux_k23 = False

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = OCDINet(
        2 * coils, 2 * coils, time_dim=128, base_ch=base_ch, k_big=(3, 9),
        T=steps, use_xattn=use_xattn, xattn_heads=xattn_heads,
        xattn_window=xattn_window, xattn_mlp=xattn_mlp,
        use_aux_k23=use_aux_k23,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    model.eval()

    test_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))
    if args.max_samples > 0:
        test_paths = test_paths[:args.max_samples]
    if not test_paths:
        raise RuntimeError(f"No .mat files found in {args.test_dir}")

    reference_dir = os.path.abspath(args.slice_ref_dir) if args.slice_ref_dir.strip() else ""
    if reference_dir and os.path.abspath(args.test_dir) == reference_dir:
        raise ValueError("--slice_ref_dir must be a separate directory from --test_dir.")

    reference_functions = None
    if reference_dir:
        from ocdi.ops.slf import (
            apply_slf_reference,
            fit_slf_reference,
            infer_mb_to_target_slf,
            infer_mb_to_target_slf_late,
            slf_rest_fusion,
        )
        reference_functions = (
            apply_slf_reference,
            fit_slf_reference,
            infer_mb_to_target_slf,
            infer_mb_to_target_slf_late,
            slf_rest_fusion,
        )

    print(f"[INFO] device={device} coils={coils} T={steps} alpha={alpha_sched}")
    print(f"[INFO] test_dir={args.test_dir} reference_data={'enabled' if reference_dir else 'not used'}")
    print(f"[INFO] test_samples={len(test_paths)} targets={targets}")

    rows = []
    start_time = time.time()
    for path in tqdm(test_paths, desc="OCDI-Test", dynamic_ncols=True):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            k1, k2, k3, kmb = load_packed_mb3_mat(path, coils)
        except Exception as exc:
            raise RuntimeError(f"Failed to load test sample {name}: {exc}") from exc

        target_slices = [k1, k2, k3]
        target_tensors = [to_2cchw(k).to(device)[None] for k in target_slices]
        kmb_tensor = to_2cchw(kmb).to(device)[None]

        scale = 1.0
        if args.amp_norm:
            scale = quantile_scale_kmb(kmb_tensor[0])
            if not math.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"Invalid normalization scale for {name}: {scale}")
            target_tensors = [value / scale for value in target_tensors]
            kmb_tensor = kmb_tensor / scale

        reference_outputs = None
        if reference_dir:
            ref_path = os.path.join(reference_dir, os.path.basename(path))
            if not os.path.isfile(ref_path):
                raise FileNotFoundError(f"Missing independently acquired slice-reference file: {ref_path}")
            apply_reference, fit_reference, infer_with_reference, infer_with_reference_late, fuse_reference = reference_functions
            reference_slices = _load_reference(ref_path, coils)
            if args.amp_norm:
                reference_slices = reference_slices / scale
            operator = fit_reference(
                reference_slices=reference_slices,
                kernel=(kernel_h, kernel_w),
                lf_lines=args.ref_lines,
                lam=float(args.lam),
                col_whiten=not args.no_col_whiten,
            )
            reference_outputs = apply_reference(kmb_tensor, operator)

        for target in targets:
            idx = target_to_idx[target]
            shift = recenter[target]
            kmb_current = apply_fov_shift_2cchw(kmb_tensor, shift_frac=shift, axis=args.shift_axis)
            gt_current = apply_fov_shift_2cchw(target_tensors[idx], shift_frac=shift, axis=args.shift_axis)
            target_reference = rest_reference = None
            if reference_outputs is not None:
                target_reference = apply_fov_shift_2cchw(reference_outputs[idx], shift_frac=shift, axis=args.shift_axis)
                rest_reference = apply_fov_shift_2cchw(
                    sum(reference_outputs[j] for j in range(3) if j != idx),
                    shift_frac=shift,
                    axis=args.shift_axis,
                )

            with torch.no_grad():
                net_prediction = infer_mb_to_k1(model, kmb_current, T=steps, alpha_sched=alpha_sched)
                proposed = net_prediction
                if target_reference is not None:
                    if args.ref_late_step:
                        proposed, _ = infer_with_reference_late(
                            model, kmb_current,
                            target_ref_2c=target_reference,
                            rest_ref_2c=rest_reference,
                            T=steps, alpha_sched=alpha_sched,
                            lf_lines=args.ref_lines,
                            slf_w=args.ref_strength, slf_pow=args.ref_power,
                            guide_delta=not args.no_guide_delta,
                            late_start=args.ref_late_start,
                            beta_max=args.ref_beta_max,
                            beta_pow=args.ref_beta_pow,
                            rest_mix=args.ref_rest_mix,
                            fuse_lines=args.ref_fuse_lines or args.ref_lines,
                            shape=args.ref_fuse_shape,
                            mode=args.ref_fuse_mode,
                            transition=args.ref_fuse_transition,
                            min_w=args.ref_min_w,
                            max_w=args.ref_max_w,
                            force_center=not args.no_ref_force_center,
                            final_adaptive=not args.ref_no_final_fuse,
                        )
                    else:
                        proposed_base = infer_with_reference(
                            model, kmb_current,
                            target_ref_2c=target_reference,
                            rest_ref_2c=rest_reference,
                            T=steps, alpha_sched=alpha_sched,
                            lf_lines=args.ref_lines,
                            slf_w=args.ref_strength,
                            slf_pow=args.ref_power,
                            guide_delta=not args.no_guide_delta,
                        )
                        if args.ref_fuse_enable:
                            proposed, _ = fuse_reference(
                                proposed_base, kmb_current, rest_reference,
                                fuse_lines=args.ref_fuse_lines or args.ref_lines,
                                shape=args.ref_fuse_shape,
                                mode=args.ref_fuse_mode,
                                transition=args.ref_fuse_transition,
                                min_w=args.ref_min_w,
                                max_w=args.ref_max_w,
                                force_center=not args.no_ref_force_center,
                            )
                        else:
                            proposed = proposed_base

            gt_image = k2img_sos_from_2cchw_matlab(gt_current[0]).detach().cpu().numpy()
            net_image = k2img_sos_from_2cchw_matlab(net_prediction[0]).detach().cpu().numpy()
            proposed_image = k2img_sos_from_2cchw_matlab(proposed[0]).detach().cpu().numpy()
            metric_args = dict(
                norm_mode=args.metric_norm,
                match_gain=not args.no_match_gain,
                do_clip=not args.no_clip,
            )
            psnr_net, ssim_net = roger_psnr_ssim_matlab_equiv(gt_image, net_image, **metric_args)
            psnr_proposed, ssim_proposed = roger_psnr_ssim_matlab_equiv(gt_image, proposed_image, **metric_args)

            if reference_outputs is not None:
                ref_image = k2img_sos_from_2cchw_matlab(target_reference[0]).detach().cpu().numpy()
                psnr_ref, ssim_ref = roger_psnr_ssim_matlab_equiv(gt_image, ref_image, **metric_args)
            else:
                psnr_ref = ssim_ref = float("nan")

            rows.append([
                name, "", "", target.upper(), f"{shift:+.6f}",
                f"{psnr_ref:.6f}", f"{ssim_ref:.6f}",
                f"{psnr_net:.6f}", f"{ssim_net:.6f}",
                f"{psnr_proposed:.6f}", f"{ssim_proposed:.6f}",
                f"{psnr_proposed - max(psnr_ref, psnr_net):.6f}" if np.isfinite(psnr_ref)
                else f"{psnr_proposed - psnr_net:.6f}",
            ])

    csv_path = os.path.join(args.out_dir, "metrics_k123_ref_net_proposed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file", "subject", "quad", "target", "recenter_shift_frac",
            "psnr_ref", "ssim_ref", "psnr_net", "ssim_net",
            "psnr_proposed", "ssim_proposed", "delta_vs_best_ref_net",
        ])
        writer.writerows(rows)

    print(f"[DONE] csv={csv_path}")
    print(f"[DONE] elapsed={time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
