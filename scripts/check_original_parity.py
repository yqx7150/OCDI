"""Synthetic comparison with the author-supplied original training script.

No real-data metrics, trained checkpoints, GPU numerics or baseline behavior
are certified. The original file is loaded for its definitions, not training.
"""
import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import random
import tempfile
import numpy as np
from scipy.io import savemat
import torch
from ocdi.data.mb3 import MB3FolderDataset
from ocdi.net import OCDINet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    loader = importlib.machinery.SourceFileLoader("original_mb3_for_parity", args.original)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    original = importlib.util.module_from_spec(spec)
    loader.exec_module(original)
    count = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "one.mat"
        rng = np.random.default_rng(2024)
        arrays = {f"K{i}": (rng.normal(size=(28, 36, 2)) + 1j * rng.normal(size=(28, 36, 2))).astype(np.complex64)
                  for i in (1, 2, 3)}
        for file_mb in (False, True):
            for normalize in (False, True):
                for mode in ("kmb", "k1"):
                    for divide in (False, True):
                        for spatial in (None, (24, 40)):
                            fields = dict(arrays)
                            if file_mb:
                                fields["KMB"] = sum(arrays.values()) * (1 + 0.01j)
                            savemat(path, fields)
                            settings = dict(coils=2, amp_norm=normalize, amp_norm_mode=mode,
                                            sum23_div_sqrt2=divide, target_hw=spatial)
                            a = original.MB3FolderDataset(directory, **settings)
                            b = MB3FolderDataset(directory, **settings)
                            random.seed(7); first = a[0]
                            random.seed(7); second = b[0]
                            for x, y in zip(first, second):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)
                            count += 1
    model_count = 0
    for attention in (False, True):
        for auxiliary in (False, True):
            settings = dict(base_ch=8, T=2, use_xattn=attention, use_aux_k23=auxiliary)
            torch.manual_seed(4); a = original.UNetSmall(4, 4, **settings).eval()
            torch.manual_seed(4); b = OCDINet(4, 4, **settings).eval()
            assert a.state_dict().keys() == b.state_dict().keys()
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
            sample = torch.randn(1, 4, 32, 40)
            with torch.inference_mode():
                torch.testing.assert_close(a(sample, torch.tensor([1])), b(sample, torch.tensor([1])), rtol=0, atol=0)
            model_count += 1
    print(f"Exact synthetic dataset parity: {count} configurations")
    print(f"Exact model state and forward-output parity: {model_count} attention/auxiliary combinations")


if __name__ == "__main__":
    main()
