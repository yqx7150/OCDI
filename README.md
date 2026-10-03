# OCDI

Official implementation of **Operator-Guided k-space Reconstruction for Simultaneous Multi-Slice MRI**.

## Structure

```text
.
├── ocdi/
│   ├── data/
│   ├── engine/
│   ├── net/
│   ├── ops/
│   └── utils/
├── train/
│   └── train.py
├── test/
│   └── test.py
├── requirements.txt
└── pyproject.toml
```

## Requirements

Python 3.9+ and PyTorch 2.1+ are recommended.

```bash
pip install -r requirements.txt
```

## Data interface

Each sample is stored as a MATLAB file containing

```text
K1
K2
K3
```

and optionally

```text
KMB
```

`K1`, `K2`, and `K3` are complex-valued multi-coil k-space arrays in the SMS frame.

## Entry points

```bash
python -m train.train --help
python -m test.test --help
```
