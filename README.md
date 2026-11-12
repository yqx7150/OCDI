# OCDI

Official implementation of **Operator-Guided k-space Reconstruction for Simultaneous Multi-Slice MRI**.

## Structure

| Directory | Description |
| --- | --- |
| `ocdi/data/` | Data loading |
| `ocdi/engine/` | Training and inference utilities |
| `ocdi/net/` | Network architecture |
| `ocdi/ops/` | Reconstruction operators |
| `ocdi/utils/` | Metrics and visualization |
| `train/` | Training entry point |
| `test/` | Evaluation entry point |
| `scripts/` | Data preparation and verification |
| `tests/` | Software tests |
| `docs/` | Data and calibration documentation |

## Requirements

Python 3.9+ and PyTorch 2.1+.

Install the package and its dependencies from the repository root:

```bash
python -m pip install .
```

## Data interface

Data are stored in MATLAB files. Multi-coil k-space arrays use complex-valued `[H, W, C]` format, where `C` denotes the number of coils and `W` is the phase-encode dimension.

| Data | Required fields |
| --- | --- |
| Packed training samples | `K1`, `K2`, `K3`; optional `KMB` |
| Test measurements | `KMB`, `mask` |
| Slice-resolved calibration references | `K1`, `K2`, `K3`, `acs_mask` |
| Ground truth for evaluation | `K1`, `K2`, `K3` |

Packed `K1`, `K2`, and `K3` are already expressed in the CAIPI-modulated SMS frame. The data loader does not apply additional CAIPI modulation.

For retrospective calibration, `slice_ref` and `ppa_ref` are simulated from the central ACS region of the corresponding single-band k-space data. Reference arrays are zero outside the declared ACS support. These are retrospectively simulated references rather than separately acquired reference scans.

Test measurements, calibration references, and evaluation ground truth are stored separately. Ground truth supplied through `--gt_dir` is used for scoring after reconstruction.

See [Data and calibration protocol](docs/DATA_PROTOCOL.md) for preparation details.

## Entry points

Display training and evaluation options:

```bash
python -m train.train --help
python -m test.test --help
```

Display retrospective data preparation options:

```bash
python -m scripts.prepare_retrospective --help
```

Run the synthetic integration check:

```bash
python -m scripts.smoke_test
```

The integration check verifies software execution using synthetic data.
