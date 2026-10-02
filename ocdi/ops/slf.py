from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _band_bounds(width: int, lines: int):
    start = max(0, width // 2 - int(lines) // 2)
    end = min(width, start + int(lines))
    return start, end


def _build_reference_system(
    kmb: np.ndarray,
    slices: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    band_start: int,
    band_end: int,
):
    mb, height, width, coils = slices.shape
    radius_h, radius_w = kernel_h // 2, kernel_w // 2
    x_min, x_max = radius_h, height - 1 - radius_h
    y_min = max(radius_w, band_start + radius_w)
    y_max = min(width - 1 - radius_w, (band_end - 1) - radius_w)

    if y_max < y_min or x_max < x_min:
        raise RuntimeError("Low-frequency band is too small for the selected kernel")

    xs = np.arange(x_min, x_max + 1, dtype=np.int32)
    ys = np.arange(y_min, y_max + 1, dtype=np.int32)
    n_samples = xs.size * ys.size
    feature_dim = coils * kernel_h * kernel_w

    x_mat = np.zeros((n_samples, feature_dim), dtype=np.complex64)
    y_mat = np.zeros((n_samples, mb * coils), dtype=np.complex64)

    row = 0
    for y in ys:
        y0, y1 = y - radius_w, y + radius_w + 1
        for x in xs:
            x0, x1 = x - radius_h, x + radius_h + 1
            patch = kmb[x0:x1, y0:y1, :]
            x_mat[row, :] = patch.reshape(-1, order="F")
            y_mat[row, :] = np.concatenate(
                [slices[s, x, y, :].reshape(-1) for s in range(mb)], axis=0
            )
            row += 1

    return x_mat, y_mat


def fit_slf_reference(
    k_slices: np.ndarray,
    kernel=(5, 5),
    lf_lines: int = 32,
    lam: float = 1e-3,
    col_whiten: bool = True,
    eps: float = 1e-12,
):
    mb, _, width, coils = k_slices.shape
    kernel_h, kernel_w = int(kernel[0]), int(kernel[1])
    kmb_ref = np.sum(k_slices, axis=0).astype(np.complex64)
    band_start, band_end = _band_bounds(width, lf_lines)

    x_mat, y_mat = _build_reference_system(
        kmb_ref,
        k_slices,
        kernel_h,
        kernel_w,
        band_start,
        band_end,
    )

    if col_whiten:
        col_scale = np.sqrt(np.sum(np.abs(x_mat) ** 2, axis=0) + eps).astype(np.float64)
        x_norm = (x_mat / col_scale[None, :]).astype(np.complex64)
    else:
        col_scale = None
        x_norm = x_mat

    x_h = np.conjugate(x_norm).T
    gram = (x_h @ x_norm).astype(np.complex64)
    rhs = (x_h @ y_mat).astype(np.complex64)
    system = gram + np.complex64(lam) * np.eye(gram.shape[0], dtype=np.complex64)
    weights = np.linalg.solve(system, rhs).astype(np.complex64)

    if col_whiten:
        weights = (weights / col_scale[:, None]).astype(np.complex64)

    operator = np.zeros(
        (mb, coils, coils, kernel_h, kernel_w), dtype=np.complex64
    )
    for out_idx in range(mb * coils):
        slice_idx = out_idx // coils
        coil_idx = out_idx % coils
        patch = weights[:, out_idx].reshape(
            (kernel_h, kernel_w, coils), order="F"
        )
        operator[slice_idx, coil_idx] = patch.transpose(2, 0, 1)

    return operator


def _complex_kernel_to_ri_weight(kernel: np.ndarray) -> torch.Tensor:
    out_channels, in_channels, kernel_h, kernel_w = kernel.shape
    real = np.real(kernel).astype(np.float32)
    imag = np.imag(kernel).astype(np.float32)

    weight = np.zeros(
        (2 * out_channels, 2 * in_channels, kernel_h, kernel_w),
        dtype=np.float32,
    )
    weight[:out_channels, :in_channels] = real
    weight[:out_channels, in_channels : 2 * in_channels] = -imag
    weight[out_channels : 2 * out_channels, :in_channels] = imag
    weight[out_channels : 2 * out_channels, in_channels : 2 * in_channels] = real
    return torch.from_numpy(weight)


@torch.no_grad()
def apply_slf_reference(kmb_2c: torch.Tensor, operator: np.ndarray) -> torch.Tensor:
    mb, _, _, kernel_h, kernel_w = operator.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    _, _, height, width = kmb_2c.shape

    outputs = []
    for slice_idx in range(mb):
        weight = _complex_kernel_to_ri_weight(operator[slice_idx]).to(kmb_2c.device)
        out = F.conv2d(kmb_2c, weight, padding=(pad_h, pad_w))
        if pad_h > 0:
            out[:, :, :pad_h, :] = 0
            out[:, :, height - pad_h :, :] = 0
        if pad_w > 0:
            out[:, :, :, :pad_w] = 0
            out[:, :, :, width - pad_w :] = 0
        outputs.append(out)

    return torch.stack(outputs, dim=0)


@torch.no_grad()
def S_LF(
    x: torch.Tensor,
    reference: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    lam: float = 1.0,
    lf_lines: int | None = None,
) -> torch.Tensor:
    r"""S_LF(X; R, W, λ) = X + λ W ⊙ (R - X)."""
    if reference is None or float(lam) <= 0:
        return x

    if weight is None:
        if lf_lines is None:
            raise ValueError("Either weight or lf_lines must be provided")
        start, end = _band_bounds(x.shape[-1], lf_lines)
        weight = torch.zeros(
            (1, 1, x.shape[-2], x.shape[-1]),
            device=x.device,
            dtype=x.dtype,
        )
        weight[..., start:end] = 1.0
    else:
        weight = weight.to(device=x.device, dtype=x.dtype)
        if weight.ndim == 2:
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.ndim == 3:
            weight = weight.unsqueeze(0)

    return x + float(lam) * weight * (reference - x)


@torch.no_grad()
def infer_mb_to_target_slf(
    net,
    kmb_2c: torch.Tensor,
    target_ref_2c: torch.Tensor | None = None,
    rest_ref_2c: torch.Tensor | None = None,
    T: int = 64,
    alpha_sched: str = "cos",
    lf_lines: int = 32,
    slf_w: float = 0.12,
    slf_pow: float = 2.5,
    guide_delta: bool = True,
):
    net.eval()
    x = kmb_2c.clone()
    alpha_fn = (
        (lambda t: t / float(T))
        if alpha_sched == "lin"
        else (lambda t: math.sin(0.5 * math.pi * t / float(T)))
    )

    for t in range(T, 0, -1):
        t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
        delta = net(x, t_tensor)
        k_hat = x - delta
        alpha_t = alpha_fn(t)
        lam_t = float(slf_w * ((1.0 - alpha_t) ** slf_pow))

        if target_ref_2c is not None and lam_t > 0:
            k_hat = S_LF(
                k_hat,
                target_ref_2c,
                lam=lam_t,
                lf_lines=lf_lines,
            )

        if guide_delta and rest_ref_2c is not None and lam_t > 0:
            delta_ref = alpha_t * rest_ref_2c
            delta = S_LF(
                delta,
                delta_ref,
                lam=lam_t,
                lf_lines=lf_lines,
            )
            k_hat = x - delta

        if t > 1:
            alpha_prev = alpha_fn(t - 1)
            rest_est = x - k_hat
            x = k_hat + (alpha_prev / (alpha_t + 1e-12)) * rest_est
        else:
            x = k_hat

    return x


def make_slf_weight(
    height: int,
    width: int,
    device,
    fuse_lines: int = 48,
    shape: str = "ky",
    mode: str = "gaussian",
    transition: float = 0.0,
    min_w: float = 0.0,
    max_w: float = 1.0,
    force_center: bool = True,
):
    fuse_lines = int(max(1, min(int(fuse_lines), max(height, width))))
    min_w = float(max(0.0, min(1.0, min_w)))
    max_w = float(max(0.0, min(1.0, max_w)))
    shape = shape.lower()
    mode = mode.lower()

    if shape == "ky":
        coord = torch.arange(width, device=device, dtype=torch.float32)
        distance = torch.abs(coord - float(width // 2))
        radius = float(fuse_lines) / 2.0
        outside = torch.clamp(distance - radius, min=0.0)
        if transition <= 0:
            transition = max(1.0, radius)

        if mode == "gaussian":
            line_weight = torch.exp(
                -(outside**2) / (2.0 * float(transition) ** 2)
            )
        elif mode == "sigmoid":
            line_weight = 1.0 / (
                1.0
                + torch.exp(
                    (distance - radius) / max(1.0, float(transition) / 4.0)
                )
            )
        elif mode == "cosine":
            u = torch.clamp(
                outside / max(1.0, float(transition)), 0.0, 1.0
            )
            line_weight = 0.5 * (1.0 + torch.cos(math.pi * u))
        else:
            raise ValueError("--slf_fuse_mode must be gaussian/sigmoid/cosine")

        if force_center:
            line_weight[distance <= radius] = 1.0
        weight = line_weight.view(1, width).expand(height, width)

    elif shape == "radial":
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        center_y, center_x = float(height // 2), float(width // 2)
        distance = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        radius = float(fuse_lines) / 2.0
        outside = torch.clamp(distance - radius, min=0.0)
        if transition <= 0:
            transition = max(1.0, radius)

        if mode == "gaussian":
            weight = torch.exp(
                -(outside**2) / (2.0 * float(transition) ** 2)
            )
        elif mode == "sigmoid":
            weight = 1.0 / (
                1.0
                + torch.exp(
                    (distance - radius) / max(1.0, float(transition) / 4.0)
                )
            )
        elif mode == "cosine":
            u = torch.clamp(
                outside / max(1.0, float(transition)), 0.0, 1.0
            )
            weight = 0.5 * (1.0 + torch.cos(math.pi * u))
        else:
            raise ValueError("--slf_fuse_mode must be gaussian/sigmoid/cosine")

        if force_center:
            weight[distance <= radius] = 1.0
    else:
        raise ValueError("--slf_fuse_shape must be ky or radial")

    weight = min_w + (max_w - min_w) * torch.clamp(weight, 0.0, 1.0)
    return weight.view(1, 1, height, width)


@torch.no_grad()
def slf_rest_fusion(
    target_pred: torch.Tensor,
    kmb: torch.Tensor,
    rest_ref: torch.Tensor,
    fuse_lines: int,
    shape: str = "ky",
    mode: str = "gaussian",
    transition: float = 0.0,
    min_w: float = 0.0,
    max_w: float = 1.0,
    force_center: bool = True,
):
    _, _, height, width = target_pred.shape
    weight = make_slf_weight(
        height,
        width,
        target_pred.device,
        fuse_lines=fuse_lines,
        shape=shape,
        mode=mode,
        transition=transition,
        min_w=min_w,
        max_w=max_w,
        force_center=force_center,
    )
    rest_hat = kmb - target_pred
    rest_fused = S_LF(rest_hat, rest_ref, weight=weight, lam=1.0)
    return kmb - rest_fused, weight


@torch.no_grad()
def infer_mb_to_target_slf_late(
    net,
    kmb_2c: torch.Tensor,
    target_ref_2c: torch.Tensor | None = None,
    rest_ref_2c: torch.Tensor | None = None,
    T: int = 64,
    alpha_sched: str = "cos",
    lf_lines: int = 32,
    slf_w: float = 0.12,
    slf_pow: float = 2.5,
    guide_delta: bool = True,
    late_start: int = 12,
    beta_max: float = 0.10,
    beta_pow: float = 2.0,
    rest_mix: float = 1.0,
    fuse_lines: int = 48,
    shape: str = "ky",
    mode: str = "gaussian",
    transition: float = 0.0,
    min_w: float = 0.0,
    max_w: float = 1.0,
    force_center: bool = True,
    final_adaptive: bool = True,
):
    net.eval()
    x = kmb_2c.clone()
    alpha_fn = (
        (lambda t: t / float(T))
        if alpha_sched == "lin"
        else (lambda t: math.sin(0.5 * math.pi * t / float(T)))
    )

    _, _, height, width = kmb_2c.shape
    weight = None
    if rest_ref_2c is not None:
        weight = make_slf_weight(
            height,
            width,
            kmb_2c.device,
            fuse_lines=fuse_lines,
            shape=shape,
            mode=mode,
            transition=transition,
            min_w=min_w,
            max_w=max_w,
            force_center=force_center,
        )

    late_start = int(max(0, min(int(late_start), int(T))))
    beta_max = float(max(0.0, min(1.0, beta_max)))
    rest_mix = float(max(0.0, min(1.0, rest_mix)))
    beta_pow = float(max(0.0, beta_pow))

    for t in range(T, 0, -1):
        t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
        delta = net(x, t_tensor)
        k_hat = x - delta
        alpha_t = float(alpha_fn(t))
        lam_t = float(slf_w * ((1.0 - alpha_t) ** slf_pow))

        if target_ref_2c is not None and lam_t > 0:
            k_hat = S_LF(
                k_hat,
                target_ref_2c,
                lam=lam_t,
                lf_lines=lf_lines,
            )

        if guide_delta and rest_ref_2c is not None and lam_t > 0:
            delta_ref = alpha_t * rest_ref_2c
            delta = S_LF(
                delta,
                delta_ref,
                lam=lam_t,
                lf_lines=lf_lines,
            )
            k_hat = x - delta

        use_anchor = (
            weight is not None
            and late_start > 0
            and t <= late_start
            and beta_max > 0
            and rest_mix > 0
        )
        if use_anchor:
            progress = float(late_start - t + 1) / float(max(1, late_start))
            beta_t = beta_max * (progress**beta_pow) * rest_mix
            beta_t = float(max(0.0, min(1.0, beta_t)))
        else:
            beta_t = 0.0

        if t > 1:
            alpha_prev = float(alpha_fn(t - 1))
            rest_net = (x - k_hat) / (alpha_t + 1e-12)
            if beta_t > 0:
                rest_used = S_LF(
                    rest_net,
                    rest_ref_2c,
                    weight=weight,
                    lam=beta_t,
                )
            else:
                rest_used = rest_net
            x = k_hat + alpha_prev * rest_used
        elif final_adaptive and weight is not None:
            rest_net_final = kmb_2c - k_hat
            rest_fused = S_LF(
                rest_net_final,
                rest_ref_2c,
                weight=weight,
                lam=1.0,
            )
            x = kmb_2c - rest_fused
        elif beta_t > 0 and weight is not None:
            rest_net_final = kmb_2c - k_hat
            rest_fused = S_LF(
                rest_net_final,
                rest_ref_2c,
                weight=weight,
                lam=beta_t,
            )
            x = kmb_2c - rest_fused
        else:
            x = k_hat

    return x, weight
