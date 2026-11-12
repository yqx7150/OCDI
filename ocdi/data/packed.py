"""Explicit MATLAB data contracts; no automatic axis guessing or test-label fallback.

Canonical arrays are complex [H, W, C], with phase encoding along W.
K1/K2/K3 in packed training files must already be in the common SMS frame.
Only load_packed_mb3_mat may synthesize a missing KMB, for retrospective
preparation/training. Test inference must use load_measurement_mat.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
import h5py


def load_mat_arrays(path):
    """Read numeric v5 MAT or numeric MATLAB v7.3 datasets.

    MATLAB-marked HDF5 datasets store axes in reverse order. Generic HDF5
    datasets without MATLAB_class retain their stored order. Cells, structs
    and object references are deliberately unsupported and raise an error.
    """
    if not h5py.is_hdf5(path):
        return {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__")}
    result = {}
    with h5py.File(path, "r") as handle:
        for key, dataset in handle.items():
            if key.startswith("#"):
                continue
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"Unsupported MAT struct/group: {key}")
            arr = np.asarray(dataset[()])
            if arr.dtype.names:
                if set(arr.dtype.names) != {"real", "imag"}:
                    raise ValueError(f"Unsupported compound field: {key}")
                arr = arr["real"] + 1j * arr["imag"]
            if arr.dtype.kind == "O":
                raise ValueError(f"Unsupported cell/object reference: {key}")
            if "MATLAB_class" in dataset.attrs and arr.ndim > 1:
                arr = arr.transpose(tuple(reversed(range(arr.ndim))))
            result[key] = arr
    return result


def _field(arrays, key, required=True):
    matches = [value for name, value in arrays.items() if name.strip().lower() == key.lower()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duplicate field: {key}")
    if not matches:
        if required:
            raise KeyError(f"Missing field: {key}")
        return None
    return matches[0]


def as_hwc(value, coils, *, layout="HWC", coil_policy="strict"):
    arr = np.asarray(value)
    if arr.ndim == 2 and coils == 1:
        arr = arr[..., None]
    if arr.ndim != 3 or layout not in ("HWC", "CHW"):
        raise ValueError("Expected numeric 3D k-space; specify HWC or CHW layout explicitly")
    if layout == "CHW":
        arr = arr.transpose(1, 2, 0)
    if arr.dtype.kind not in "fciub":
        raise ValueError(f"Unsupported k-space dtype: {arr.dtype}")
    arr = arr.astype(np.complex64)
    if min(arr.shape) <= 0 or not np.isfinite(arr).all():
        raise ValueError("k-space must be nonempty and finite")
    if coils <= 0 or coil_policy not in ("strict", "legacy"):
        raise ValueError("Invalid coil count or coil policy")
    if arr.shape[-1] != coils:
        if coil_policy == "strict":
            raise ValueError(f"Expected {coils} coils, got {arr.shape[-1]}; no silent conversion")
        # Original MB3R1.PY training behavior: truncate or zero-pad coil channels.
        arr = arr[..., :coils]
        if arr.shape[-1] < coils:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, coils - arr.shape[-1])))
    return np.ascontiguousarray(arr)


def load_packed_mb3_mat(path, coils, *, layout="HWC", coil_policy="strict"):
    """Return (K1, K2, K3, KMB), for packed TRAINING/offline retrospective data.

    A supplied KMB is preserved. A missing KMB is synthesized by summing the
    three already-modulated slices. This fallback is never used for inference.
    No phase ramps, normalization or spatial cropping are applied here.
    """
    arrays = load_mat_arrays(path)
    slices = [as_hwc(_field(arrays, f"K{i}"), coils, layout=layout,
                     coil_policy=coil_policy) for i in (1, 2, 3)]
    if len({x.shape for x in slices}) != 1:
        raise ValueError("K1/K2/K3 shapes differ")
    measured = _field(arrays, "KMB", required=False)
    kmb = (sum(slices) if measured is None else
           as_hwc(measured, coils, layout=layout, coil_policy=coil_policy))
    if kmb.shape != slices[0].shape:
        raise ValueError("KMB and slice shapes differ")
    return (*slices, np.ascontiguousarray(kmb, dtype=np.complex64))


def load_measurement_mat(path, coils):
    """Read inference-only KMB and declared binary mask; reject packed labels."""
    arrays = load_mat_arrays(path)
    if any(name.strip().lower() in ("k1", "k2", "k3") for name in arrays):
        raise ValueError("Test inputs must contain measurements only; move K1/K2/K3 to --gt_dir")
    kmb = as_hwc(_field(arrays, "KMB"), coils)
    mask = _field(arrays, "mask")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != kmb.shape[:2] or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must be binary [H,W]")
        mask = mask.astype(bool)
        if np.any(kmb[~mask] != 0):
            raise ValueError("Nonzero measurement outside the declared acquisition mask")
    return kmb, mask


def central_acs_mask(height, width, lines):
    if not 0 < lines <= width:
        raise ValueError("ACS lines must be between 1 and W")
    start = width // 2 - lines // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[:, start:start + lines] = True
    return mask


def load_acs_reference_mat(path, coils, lines):
    """Read zero-filled slice-resolved ACS. Reject nonzero outer reference data."""
    arrays = load_mat_arrays(path)
    slices = [as_hwc(_field(arrays, f"K{i}"), coils) for i in (1, 2, 3)]
    if len({x.shape for x in slices}) != 1:
        raise ValueError("ACS slice shapes differ")
    mask = np.asarray(_field(arrays, "acs_mask"))
    expected = central_acs_mask(*slices[0].shape[:2], lines)
    if mask.shape != expected.shape or not np.array_equal(mask, expected):
        raise ValueError("acs_mask does not match the declared central ACS band")
    if any(np.any(x[~expected] != 0) for x in slices):
        raise ValueError("ACS reference contains nonzero values outside its declared support")
    return np.stack(slices)
