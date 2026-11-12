# OCDI — MB3R1 packaging and data-interface release candidate

Implementation associated with **Operator-Guided k-space Reconstruction for Simultaneous Multi-Slice MRI**.

This candidate repairs missing data modules and makes the MB3R1 data boundary explicit. It is not a verified release of every configuration or baseline in the paper. Stage-U, MB4, the historical raw-data conversion, actual split lists, paper checkpoints and baseline adapters are not supplied by this candidate. See [release checklist](docs/RELEASE_CHECKLIST.md) before promoting a release.

## Installation

Use a fresh clone and a new Python environment. The CPU check uses Python 3.12, PyTorch 2.5.1 CPU and `constraints-cpu-tested.txt`. This does not specify the paper's CUDA environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-cpu-tested.txt .
python -m pip check
python -m train.train --help
python -m test.test --help
python -m unittest discover -s tests -v
python -m scripts.smoke_test
```

The smoke check uses synthetic complex arrays, one tiny training update and three guided evaluation runs. It checks that predictions are unchanged when scoring labels are altered or absent. Its metrics are not paper results. Also run installed entry points outside the checkout.

For GPU experiments, install and record the experiment's PyTorch/CUDA environment separately. The CPU smoke configuration and training defaults are not paper configurations.

## Data contracts

Numeric k-space is complex `[H,W,C]`; `W` is the phase-encode axis. Supported formats are numeric v5 MAT and numeric HDF5 datasets, including MATLAB-marked axis order and `real`/`imag` compound values. Arbitrary MATLAB structs/cells/object references are unsupported.

| Role | Required fields | Allowed use |
| --- | --- | --- |
| Packed training/offline preparation | `K1`, `K2`, `K3`; optional `KMB` | Supervised training and retrospective generation |
| Test measurements | `KMB`, binary `[H,W]` `mask` | Reconstruction input; packed labels are rejected |
| Slice-resolved central ACS | Zero-filled `K1`, `K2`, `K3`, binary `acs_mask` | Scan-specific calibration |
| Scoring ground truth | `K1`, `K2`, `K3` | Optional scoring after prediction |

Packed slices must already share the CAIPI-modulated SMS frame. No phase ramps are added by loading/preparation. A missing packed `KMB` is synthesized by summation only for training/offline preparation; measurement loading never synthesizes inputs from scoring labels.

The restored dataset follows supplied `MB3R1.PY` for interpolation, timestep sampling, normalization and optional K23 scaling. Training retains original coil truncation/zero-padding; inference rejects mismatched coils. Silent axis guessing is removed. Convert CHW data explicitly to HWC.

## Retrospective preparation

The utility partitions measurements, slice-resolved ACS and scoring labels. It is not the original fastMRI-to-packed conversion. First verify subject splits, CAIPI convention, coils and the historical acquisition mask.

```bash
python -m scripts.audit_splits --manifest /path/to/splits.csv --verify_files
python -m scripts.prepare_retrospective --packed_dir /path/to/test_packed --out_dir /path/to/prepared --coils 16 --acs_lines 32 --R 1 --confirm_sms_frame
```

Outputs: `measurements/`, `slice_ref/`, `ppa_ref/`, `ground_truth/`, `preparation.json`. References are zero outside ACS. Both reference directories expose the same allowed single-band ACS samples; each baseline's representation, known phase demodulation, coil maps and calibration still need verification.

These references are retrospectively simulated, not separately acquired scans. Slice-resolved ACS is additional information beyond collapsed multiband ACS. Calibration overhead is not included in nominal in-plane R. `--include_mb_acs` explicitly adds ACS to the multiband acquisition mask and is off by default. Confirm the historical mask before changing it.

## MB3R1 evaluation

This is a command template, not a certified paper configuration. Use the correct paper checkpoint and its recorded architecture. Earlier best checkpoints may omit `args`; supply their exact settings.

```bash
python -m test.test --ckpt /path/to/best_model.pt --test_dir /path/to/prepared/measurements --slice_ref_dir /path/to/prepared/slice_ref --gt_dir /path/to/prepared/ground_truth --out_dir /path/to/results --device cpu --coils 16 --base_ch 128 --T 64 --amp_norm --ref_lines 32 --kernel 9,9
```

Remove `--gt_dir` for inference without labels. Omitting `--slice_ref_dir` runs the network-only variant. Add `--xattn`/`--use_aux_k23` if required by a checkpoint without metadata. Loading is strict. This entry point refuses missing-line masks: R>1 needs the separately validated in-plane completion/Stage-U implementation.

Predictions are saved as `{sample}_{target}.npz`, in the target-recentered frame with `recenter_shift_frac`, rescaled to input units. Metrics go to `metrics_k123_ref_net_proposed.csv` and are NaN without GT. Legacy scoring uses GT-based magnitude scaling, gain matching and clipping by default, after prediction. Disclose and match these operations across methods; `--no_match_gain` and `--no_clip` disable the respective operations and may change reported metrics.

## Scientific scope

Read [DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md). General learned weights and hyperparameters must use training/validation subjects only. Test-specific calibration is restricted to declared ACS. Full test labels are permitted for offline simulation and scoring, not unrestricted reconstruction or model selection.

SG, SPSG, SENSE, RAKI, KDM, ROGER and FM sources were not supplied for this candidate. Their actual dependencies and paper metrics require separate verification. Respect data-access/redistribution terms and third-party code licenses. No clinical data is bundled.
