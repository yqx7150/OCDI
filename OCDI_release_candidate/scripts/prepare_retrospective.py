"""Offline partitioning of ALREADY CAIPI-modulated packed MB3 data.

This utility formalizes an information budget; it does not reproduce or certify
the historical preprocessing, generate CAIPI ramps, or implement baselines.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import savemat
from ocdi.data.packed import load_packed_mb3_mat, central_acs_mask


def prepare_one(source, output, *, coils, acs_lines, acceleration=1, offset=0,
                include_mb_acs=False):
    source, output = Path(source), Path(output)
    if acceleration < 1 or not 0 <= offset < acceleration:
        raise ValueError("R must be positive and offset must lie in [0,R)")
    k1, k2, k3, full_mb = load_packed_mb3_mat(source, coils)
    height, width, _ = k1.shape
    acs = central_acs_mask(height, width, acs_lines)
    measured = np.zeros((height, width), dtype=bool)
    measured[:, offset::acceleration] = True
    if include_mb_acs:
        measured |= acs
    payloads = {
        "measurements": {"KMB": full_mb * measured[..., None], "mask": measured.astype(np.uint8)},
        "ground_truth": {"K1": k1, "K2": k2, "K3": k3},
        "slice_ref": {"K1": k1 * acs[..., None], "K2": k2 * acs[..., None],
                      "K3": k3 * acs[..., None], "acs_mask": acs.astype(np.uint8)},
    }
    # Same permitted SB-ACS samples; the method-specific processing is separate.
    payloads["ppa_ref"] = payloads["slice_ref"]
    for kind in payloads:
        if (output / kind / source.name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / kind / source.name}")
    for kind, payload in payloads.items():
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        savemat(folder / source.name, payload, do_compression=True)
    return {
        "sample_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape_hwc": [height, width, coils], "phase_encode_axis": "W",
        "acs_lines": acs_lines, "acs_start_zero_based": width // 2 - acs_lines // 2,
        "acs_stop_exclusive": width // 2 - acs_lines // 2 + acs_lines,
        "R_nominal_in_plane": acceleration, "mask_offset": offset,
        "include_mb_acs": include_mb_acs,
        "R_effective_in_plane_measurements_only": float(width / measured[0].sum()),
        "slice_reference_origin": "retrospectively simulated slice-resolved SB ACS",
        "ppa_reference_origin": "same permitted slice-resolved SB ACS support",
        "caipi_ramp_applied_by_this_utility": False,
        "separate_calibration_scan_time_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--coils", type=int, required=True)
    parser.add_argument("--acs_lines", type=int, default=32)
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include_mb_acs", action="store_true")
    parser.add_argument("--confirm_sms_frame", action="store_true",
                        help="Confirm packed K1/K2/K3 already share the correct CAIPI/SMS frame")
    args = parser.parse_args()
    if not args.confirm_sms_frame:
        parser.error("Confirm the original preprocessing and pass --confirm_sms_frame")
    files = sorted(Path(args.packed_dir).glob("*.mat"))
    if not files:
        parser.error("No packed .mat samples found")
    output = Path(args.out_dir)
    if output.resolve() == Path(args.packed_dir).resolve():
        parser.error("Output must be separate from packed source")
    if (output / "preparation.json").exists():
        parser.error("Output already has a preparation record; use a new directory")
    records = [prepare_one(path, output, coils=args.coils, acs_lines=args.acs_lines,
                           acceleration=args.R, offset=args.offset,
                           include_mb_acs=args.include_mb_acs) for path in files]
    (output / "preparation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} samples; inspect {output / 'preparation.json'}")


if __name__ == "__main__":
    main()
