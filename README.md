# OCDI

Official implementation of **Operator-Guided k-space Reconstruction for Simultaneous Multi-Slice MRI**.

## Structure

| Path | Description |
| --- | --- |
| `ocdi/data/` | Data loading |
| `ocdi/engine/` | Training and inference utilities |
| `ocdi/net/` | Network architecture |
| `ocdi/ops/` | Reconstruction operators |
| `ocdi/utils/` | Metrics and visualization |
| `train/train.py` | Training entry point |
| `test/test.py` | Evaluation entry point |
| `scripts/` | Data preparation utilities |
| `tests/` | Software tests |

## Requirements

Python 3.9+ and PyTorch 2.1+.

```bash
git clone https://github.com/yqx7150/OCDI.git
cd OCDI
python -m pip install .
```

## Data interface

Data are stored as MATLAB files with complex-valued multi-coil k-space arrays in `[H, W, C]` format, where `C` denotes the number of coils.

Training samples contain `K1`, `K2`, `K3`, and optionally `KMB`. The individual slices are already expressed in the CAIPI-modulated SMS frame.

Test measurements contain `KMB` and a binary sampling `mask`. Calibration references contain `K1`, `K2`, `K3`, and `acs_mask`, with values outside the declared ACS region set to zero.

Calibration references are retrospectively simulated from the central ACS region of the corresponding single-band k-space data.

Ground truth containing `K1`, `K2`, and `K3` can be supplied separately through `--gt_dir` for evaluation.

## Entry points

Training options:

```bash
python -m train.train --help
```

Evaluation options:

```bash
python -m test.test --help
```

Data preparation options:

```bash
python -m scripts.prepare_retrospective --help
```
