"""Synthetic CPU integration check; does NOT reproduce paper metrics.

Runs one training update with a distinct validation phantom, loads its selected
checkpoint, and runs guided evaluation with/without/with perturbed scoring GT.
Reconstruction must be identical in all three cases.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.io import savemat
from .prepare_retrospective import prepare_one


def phantom(path, seed):
    rng = np.random.default_rng(seed)
    fields = {f"K{i}": (rng.normal(size=(32, 40, 2)) + 1j * rng.normal(size=(32, 40, 2))).astype(np.complex64)
              for i in (1, 2, 3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, fields)
    return fields


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="ocdi-smoke-") as directory:
        root = Path(directory)
        phantom(root / "train/train.mat", 1)
        phantom(root / "val/val.mat", 2)
        gt = phantom(root / "test/sample.mat", 3)
        data = root / "prepared"
        prepare_one(root / "test/sample.mat", data, coils=2, acs_lines=12)
        cmd = [sys.executable, "-m", "train.train", "--train_dir", str(root / "train"),
               "--val_dir", str(root / "val"), "--val_samples", "1", "--batch_size", "1",
               "--epochs", "1", "--num_workers", "0", "--coils", "2", "--base_ch", "8",
               "--T", "2", "--xattn", "--use_aux_k23", "--amp_norm", "--save_dir", str(root / "ckpt"),
               "--viz_dir", str(root / "viz")]
        subprocess.run(cmd, env=env, check=True)
        altered = root / "altered_gt"; altered.mkdir()
        savemat(altered / "sample.mat", {key: value * (2 + 3j) for key, value in gt.items()})
        outputs = []
        for tag, gt_dir in (("normal_gt", data / "ground_truth"), ("altered_gt", altered), ("no_gt", None)):
            out = root / tag
            cmd = [sys.executable, "-m", "test.test", "--ckpt", str(root / "ckpt/best_model.pt"),
                   "--test_dir", str(data / "measurements"), "--out_dir", str(out),
                   "--slice_ref_dir", str(data / "slice_ref"), "--ref_lines", "12", "--kernel", "3,3",
                   "--device", "cpu", "--targets", "k1,k2,k3", "--amp_norm"]
            if gt_dir is not None:
                cmd += ["--gt_dir", str(gt_dir)]
            subprocess.run(cmd, env=env, check=True)
            outputs.append(np.stack([np.load(out / f"sample_{target}.npz")["K_pred"] for target in ("k1", "k2", "k3")]))
        for result in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], result)
        print(json.dumps({"status": "passed", "synthetic_training_updates": 1,
                          "guided_evaluation_runs": 3,
                          "prediction_invariant_to_scoring_gt": True,
                          "paper_metrics_verified": False}, indent=2))


if __name__ == "__main__":
    main()
