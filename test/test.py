from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ocdi.data.packed import load_packed_mb3_mat
from ocdi.engine.inference import infer_mb_to_k1
from ocdi.net import OCDINet
from ocdi.ops.caipi import apply_fov_shift_2cchw
from ocdi.ops.complex import quantile_scale_kmb, to_2cchw
from ocdi.ops.slf import (
    apply_slf_reference,
    fit_slf_reference,
    infer_mb_to_target_slf,
    infer_mb_to_target_slf_late,
    slf_rest_fusion,
)
from ocdi.utils.metrics import k2img_sos_from_2cchw_matlab, roger_psnr_ssim_matlab_equiv
from ocdi.utils.visualization import (
    array_to_gray01,
    get_panel_scale_from_gt,
    save_clean_panel,
    save_error_panel,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OCDI evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
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
        help="Stored MB3 CAIPI shifts for K1,K2,K3.",
    )

    p.add_argument("--lf_lines", type=int, default=32)
    p.add_argument("--kernel", type=str, default="5,5")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    p.add_argument("--no_col_whiten", action="store_true")
    p.add_argument("--no_slf_ref", action="store_true")

    p.add_argument("--slf_w", type=float, default=0.12)
    p.add_argument("--slf_pow", type=float, default=2.5)
    p.add_argument("--no_guide_delta", action="store_true")

    p.add_argument("--slf_fuse_enable", action="store_true")
    p.add_argument("--slf_fuse_lines", type=int, default=0)
    p.add_argument("--slf_fuse_shape", type=str, default="ky", choices=["ky", "radial"])
    p.add_argument(
        "--slf_fuse_mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "sigmoid", "cosine"],
    )
    p.add_argument("--slf_fuse_transition", type=float, default=0.0)
    p.add_argument("--slf_min_w", type=float, default=0.0)
    p.add_argument("--slf_max_w", type=float, default=1.0)
    p.add_argument("--no_slf_force_center", action="store_true")
    p.add_argument("--save_slf_weight_map", action="store_true")

    p.add_argument("--slf_late_step_enable", action="store_true")
    p.add_argument("--slf_late_start", type=int, default=12)
    p.add_argument("--slf_beta_max", type=float, default=0.10)
    p.add_argument("--slf_beta_pow", type=float, default=2.0)
    p.add_argument("--slf_rest_mix", type=float, default=1.0)
    p.add_argument("--slf_no_final_fuse", action="store_true")

    p.add_argument("--metric_norm", type=str, default="max", choices=["max", "p99"])
    p.add_argument("--no_match_gain", action="store_true")
    p.add_argument("--no_clip", action="store_true")

    p.add_argument("--export_panels", action="store_true")
    p.add_argument("--panel_dir", type=str, default="")
    p.add_argument("--panel_format", type=str, default="png", choices=["png", "tiff", "pdf", "svg"])
    p.add_argument("--panel_dpi", type=int, default=300)
    p.add_argument("--panel_norm", type=str, default="gt_p99", choices=["gt_p99", "gt_max", "self_p99"])
    p.add_argument("--panel_p", type=float, default=99.0)
    p.add_argument("--panel_text", action="store_true")
    p.add_argument("--err_mode", type=str, default="abs", choices=["abs", "rel"])
    p.add_argument("--err_vmin", type=float, default=0.0)
    p.add_argument("--err_vmax", type=float, default=0.01)
    p.add_argument("--err_cbar_last", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    panel_root = args.panel_dir.strip() or os.path.join(args.out_dir, "panels_k123")
    os.makedirs(panel_root, exist_ok=True)

    slf_reference_enabled = not args.no_slf_ref
    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()] or ["k1", "k2", "k3"]
    target_to_idx = {"k1": 0, "k2": 1, "k3": 2}
    invalid_targets = [t for t in targets if t not in target_to_idx]
    if invalid_targets:
        raise ValueError(f"Invalid targets: {invalid_targets}")

    shift_fracs = tuple(float(x.strip()) for x in args.shift_fracs.split(","))
    if len(shift_fracs) != 3:
        raise ValueError("--shift_fracs must contain three comma-separated values for MB3.")
    recenter = {f"k{i + 1}": -shift_fracs[i] for i in range(3)}
    kH, kW = [int(x.strip()) for x in args.kernel.split(",")]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    coils = int(ck_args.get("coils", args.coils))
    T = int(ck_args.get("T", args.T))
    alpha_sched = str(ck_args.get("alpha_sched", args.alpha_sched))
    base_ch = int(ck_args.get("base_ch", args.base_ch))
    use_xattn = bool(ck_args.get("xattn", False))
    xattn_heads = int(ck_args.get("xattn_heads", args.xattn_heads))
    xattn_window = int(ck_args.get("xattn_window", args.xattn_window))
    xattn_mlp = float(ck_args.get("xattn_mlp", args.xattn_mlp))
    use_aux_k23 = bool(ck_args.get("use_aux_k23", False))

    if args.xattn:
        use_xattn = True
    if args.no_xattn:
        use_xattn = False
    if args.use_aux_k23:
        use_aux_k23 = True
    if args.no_aux_k23:
        use_aux_k23 = False

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    print(
        f"[INFO] device={device} coils={coils} T={T} alpha={alpha_sched} "
        f"base_ch={base_ch} xattn={use_xattn} aux={use_aux_k23}"
    )
    print(f"[INFO] test_dir={args.test_dir} stored_shift_fracs={shift_fracs} recenter={recenter}")
    print(
        f"[INFO] slf_ref={slf_reference_enabled} lf_lines={args.lf_lines} "
        f"kernel={kH}x{kW} lam={args.lam:g}"
    )
    print(
        f"[INFO] slf_fuse={args.slf_fuse_enable} shape={args.slf_fuse_shape} "
        f"mode={args.slf_fuse_mode} lines={args.slf_fuse_lines or args.lf_lines}"
    )
    print(
        f"[INFO] slf_late_step={args.slf_late_step_enable} start={args.slf_late_start} "
        f"beta_max={args.slf_beta_max:g} beta_pow={args.slf_beta_pow:g} "
        f"rest_mix={args.slf_rest_mix:g} final_fuse={not args.slf_no_final_fuse}"
    )

    net = OCDINet(
        2 * coils,
        2 * coils,
        time_dim=128,
        base_ch=base_ch,
        k_big=(3, 9),
        T=T,
        use_xattn=use_xattn,
        xattn_heads=xattn_heads,
        xattn_window=xattn_window,
        xattn_mlp=xattn_mlp,
        use_aux_k23=use_aux_k23,
    ).to(device)

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    net.eval()

    paths = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise RuntimeError(f"No .mat files found in {args.test_dir}")
    print(f"[INFO] packed MB3 files={len(paths)} targets={targets}")

    rows = []
    t0 = time.time()
    weight_saved = False

    for path in tqdm(paths, desc="OCDI-Test", dynamic_ncols=True):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            K1, K2, K3, KMB = load_packed_mb3_mat(path, coils)
        except Exception as exc:
            print(f"[WARN] Failed loading {name}: {exc}")
            continue

        shifted = [K1, K2, K3]
        K_slices_np = np.stack(shifted, axis=0).astype(np.complex64)
        K0 = [to_2cchw(K).to(device)[None] for K in shifted]
        KMB_2c0 = to_2cchw(KMB).to(device)[None]

        if args.amp_norm:
            scale = quantile_scale_kmb(KMB_2c0[0])
            K0 = [K / scale for K in K0]
            KMB_2c0 = KMB_2c0 / scale
            K_slices_np_for_ref = K_slices_np / scale
        else:
            K_slices_np_for_ref = K_slices_np

        if slf_reference_enabled:
            try:
                operator_weights = fit_slf_reference(
                    K_slices_np_for_ref,
                    kernel=(kH, kW),
                    lf_lines=args.lf_lines,
                    lam=float(args.lam),
                    col_whiten=not args.no_col_whiten,
                )
                K_ref_all = apply_slf_reference(KMB_2c0, operator_weights)
            except Exception as exc:
                print(f"[WARN] SLF reference failed for {name}: {exc}")
                K_ref_all = None
        else:
            K_ref_all = None

        for target in targets:
            idx = target_to_idx[target]
            shift_recenter = recenter[target]
            KMB_2c = apply_fov_shift_2cchw(KMB_2c0, shift_frac=shift_recenter, axis=args.shift_axis)
            Kt_gt = apply_fov_shift_2cchw(K0[idx], shift_frac=shift_recenter, axis=args.shift_axis)

            if K_ref_all is not None:
                Kt_ref0 = K_ref_all[idx]
                Krest_ref0 = sum(K_ref_all[j] for j in range(3) if j != idx)
                Kt_ref = apply_fov_shift_2cchw(Kt_ref0, shift_frac=shift_recenter, axis=args.shift_axis)
                Krest_ref = apply_fov_shift_2cchw(Krest_ref0, shift_frac=shift_recenter, axis=args.shift_axis)
            else:
                Kt_ref = None
                Krest_ref = None

            try:
                with torch.no_grad():
                    Kt_net = infer_mb_to_k1(net, KMB_2c, T=T, alpha_sched=alpha_sched)

                    if Kt_ref is not None:
                        if args.slf_late_step_enable:
                            Kt_prop, weight_map = infer_mb_to_target_slf_late(
                                net,
                                KMB_2c,
                                target_ref_2c=Kt_ref,
                                rest_ref_2c=Krest_ref,
                                T=T,
                                alpha_sched=alpha_sched,
                                lf_lines=args.lf_lines,
                                slf_w=args.slf_w,
                                slf_pow=args.slf_pow,
                                guide_delta=not args.no_guide_delta,
                                late_start=args.slf_late_start,
                                beta_max=args.slf_beta_max,
                                beta_pow=args.slf_beta_pow,
                                rest_mix=args.slf_rest_mix,
                                fuse_lines=args.slf_fuse_lines or args.lf_lines,
                                shape=args.slf_fuse_shape,
                                mode=args.slf_fuse_mode,
                                transition=args.slf_fuse_transition,
                                min_w=args.slf_min_w,
                                max_w=args.slf_max_w,
                                force_center=not args.no_slf_force_center,
                                final_adaptive=not args.slf_no_final_fuse,
                            )
                        else:
                            Kt_prop_base = infer_mb_to_target_slf(
                                net,
                                KMB_2c,
                                target_ref_2c=Kt_ref,
                                rest_ref_2c=Krest_ref,
                                T=T,
                                alpha_sched=alpha_sched,
                                lf_lines=args.lf_lines,
                                slf_w=args.slf_w,
                                slf_pow=args.slf_pow,
                                guide_delta=not args.no_guide_delta,
                            )

                            if args.slf_fuse_enable:
                                Kt_prop, weight_map = slf_rest_fusion(
                                    Kt_prop_base,
                                    KMB_2c,
                                    Krest_ref,
                                    fuse_lines=args.slf_fuse_lines or args.lf_lines,
                                    shape=args.slf_fuse_shape,
                                    mode=args.slf_fuse_mode,
                                    transition=args.slf_fuse_transition,
                                    min_w=args.slf_min_w,
                                    max_w=args.slf_max_w,
                                    force_center=not args.no_slf_force_center,
                                )
                            else:
                                Kt_prop = Kt_prop_base
                                weight_map = None

                        if args.save_slf_weight_map and not weight_saved and weight_map is not None:
                            plt.figure(figsize=(4, 3))
                            plt.imshow(weight_map[0, 0].detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
                            plt.colorbar()
                            plt.title(r"$S_{\mathrm{LF}}$ weight")
                            plt.tight_layout()
                            plt.savefig(os.path.join(args.out_dir, "slf_weight_map.png"), dpi=200)
                            plt.close()
                            weight_saved = True
                    else:
                        Kt_prop = Kt_net
            except Exception as exc:
                print(f"[WARN] Inference failed for {name} {target}: {exc!r}")
                continue

            img_gt = k2img_sos_from_2cchw_matlab(Kt_gt[0]).detach().cpu().numpy()
            img_net = k2img_sos_from_2cchw_matlab(Kt_net[0]).detach().cpu().numpy()
            img_prop = k2img_sos_from_2cchw_matlab(Kt_prop[0]).detach().cpu().numpy()

            if Kt_ref is not None:
                img_ref = k2img_sos_from_2cchw_matlab(Kt_ref[0]).detach().cpu().numpy()
                ps_ref, ss_ref = roger_psnr_ssim_matlab_equiv(
                    img_gt,
                    img_ref,
                    norm_mode=args.metric_norm,
                    match_gain=not args.no_match_gain,
                    do_clip=not args.no_clip,
                )
            else:
                img_ref = np.zeros_like(img_gt)
                ps_ref = ss_ref = float("nan")

            ps_net, ss_net = roger_psnr_ssim_matlab_equiv(
                img_gt,
                img_net,
                norm_mode=args.metric_norm,
                match_gain=not args.no_match_gain,
                do_clip=not args.no_clip,
            )
            ps_prop, ss_prop = roger_psnr_ssim_matlab_equiv(
                img_gt,
                img_prop,
                norm_mode=args.metric_norm,
                match_gain=not args.no_match_gain,
                do_clip=not args.no_clip,
            )

            rows.append(
                [
                    name,
                    "",
                    "",
                    target.upper(),
                    f"{shift_recenter:+.6f}",
                    f"{ps_ref:.6f}",
                    f"{ss_ref:.6f}",
                    f"{ps_net:.6f}",
                    f"{ss_net:.6f}",
                    f"{ps_prop:.6f}",
                    f"{ss_prop:.6f}",
                    f"{ps_prop - max(ps_ref, ps_net):.6f}",
                ]
            )

            if args.export_panels:
                panel_dir = os.path.join(panel_root, target.upper(), name)
                os.makedirs(panel_dir, exist_ok=True)
                panel_scale = get_panel_scale_from_gt(img_gt, args.panel_norm, args.panel_p)
                normalize = lambda x: array_to_gray01(x, scale=panel_scale, p=args.panel_p)
                fmt = args.panel_format.lower()

                def metric_text(psnr, ssim):
                    return f"{psnr:.2f}/{ssim:.2f}" if np.isfinite(psnr) else ""

                save_clean_panel(
                    normalize(img_gt),
                    os.path.join(panel_dir, f"GT_main.{fmt}"),
                    add_text=False,
                    dpi=args.panel_dpi,
                )
                save_clean_panel(
                    normalize(img_ref),
                    os.path.join(panel_dir, f"REF_main.{fmt}"),
                    text=metric_text(ps_ref, ss_ref),
                    add_text=args.panel_text,
                    dpi=args.panel_dpi,
                )
                save_clean_panel(
                    normalize(img_net),
                    os.path.join(panel_dir, f"NET_main.{fmt}"),
                    text=metric_text(ps_net, ss_net),
                    add_text=args.panel_text,
                    dpi=args.panel_dpi,
                )
                save_clean_panel(
                    normalize(img_prop),
                    os.path.join(panel_dir, f"Proposed_main.{fmt}"),
                    text=metric_text(ps_prop, ss_prop),
                    add_text=args.panel_text,
                    dpi=args.panel_dpi,
                )

                epsv = 1e-12

                def error_map(pred):
                    if args.err_mode == "rel":
                        return np.abs(pred - img_gt) / (np.abs(img_gt) + epsv)
                    return np.abs(normalize(pred) - normalize(img_gt))

                save_error_panel(
                    error_map(img_ref),
                    os.path.join(panel_dir, f"REF_err.{fmt}"),
                    vmin=args.err_vmin,
                    vmax=args.err_vmax,
                    dpi=args.panel_dpi,
                )
                save_error_panel(
                    error_map(img_net),
                    os.path.join(panel_dir, f"NET_err.{fmt}"),
                    vmin=args.err_vmin,
                    vmax=args.err_vmax,
                    dpi=args.panel_dpi,
                )
                save_error_panel(
                    error_map(img_prop),
                    os.path.join(panel_dir, f"Proposed_err.{fmt}"),
                    vmin=args.err_vmin,
                    vmax=args.err_vmax,
                    dpi=args.panel_dpi,
                    add_cbar=args.err_cbar_last,
                )

    csv_path = os.path.join(args.out_dir, "metrics_k123_ref_net_proposed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "subject",
                "quad",
                "target",
                "recenter_shift_frac",
                "psnr_ref",
                "ssim_ref",
                "psnr_net",
                "ssim_net",
                "psnr_proposed",
                "ssim_proposed",
                "delta_vs_best_ref_net",
            ]
        )
        writer.writerows(rows)

    summary_path = os.path.join(args.out_dir, "metrics_k123_summary_meanpsnr.csv")
    if not rows:
        print("[WARN] No successful reconstructions were produced.")
        print(f"[DONE] csv={csv_path}")
        print(f"[DONE] elapsed={time.time() - t0:.1f}s")
        return

    by_target = {}
    for row in rows:
        target = row[3]
        values = [float(row[5]), float(row[7]), float(row[9]), float(row[11])]
        by_target.setdefault(target, []).append(values)
        by_target.setdefault("ALL", []).append(values)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target",
                "psnr_ref_mean",
                "psnr_net_mean",
                "psnr_proposed_mean",
                "delta_vs_best_ref_net_mean",
                "n",
            ]
        )
        for target in sorted(k for k in by_target if k != "ALL") + ["ALL"]:
            arr = np.array(by_target[target], dtype=np.float64)
            writer.writerow(
                [
                    target,
                    f"{np.nanmean(arr[:, 0]):.6f}",
                    f"{np.nanmean(arr[:, 1]):.6f}",
                    f"{np.nanmean(arr[:, 2]):.6f}",
                    f"{np.nanmean(arr[:, 3]):.6f}",
                    arr.shape[0],
                ]
            )
            print(
                f"[SUMMARY] {target}: REF={np.nanmean(arr[:, 0]):.4f} "
                f"NET={np.nanmean(arr[:, 1]):.4f} "
                f"Proposed={np.nanmean(arr[:, 2]):.4f} "
                f"delta={np.nanmean(arr[:, 3]):.4f} n={arr.shape[0]}"
            )

    print(f"[DONE] csv={csv_path}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
