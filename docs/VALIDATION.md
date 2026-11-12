# Local validation of the MB3R1 candidate

Date: 2026-11-12. Starting public commit: `50874310cbb2dc876df49d086b63742af5c06a3a`.

The uploaded original `MB3R1.PY` is preserved byte-for-byte at `reference/MB3R1_original.py`.
SHA256: `becd7c5328aeef9808f330a57dd818df74dc25b346b95379a6ee1c9616d7b215`.

## Environment and installation

Linux, Python 3.12.14, torch 2.5.1+cpu, numpy 1.26.4, scipy 1.14.1, h5py 3.12.1, matplotlib 3.9.2, tqdm 4.67.1.

A new virtual environment was created, the non-editable wheel installed, and all installed-entry checks run outside the source checkout with PYTHONPATH removed. The imported ocdi module was in that environment's site-packages. Package metadata and __version__ both reported 0.1.1rc1. This verifies CPU packaging, not a historical paper training environment.

## Passed

| Check | Result |
| --- | --- |
| Build wheel and sdist | Success; data and documented entry packages present |
| sdist contents | Protocol, original reference source, data tests and constraints included |
| pip check | No broken requirements |
| Installed train/test help | Both return code 0 outside checkout |
| Installed prepare/split utilities | Both return code 0 outside checkout |
| Data contracts | 14 unittest tests passed |
| Original/restored dataset | Exact tensor equality for 32 synthetic HWC configurations |
| Original/public model | Exact state dictionaries and forward outputs for 4 attention/auxiliary combinations |
| Synthetic training | One CPU update; separate synthetic validation sample; selected checkpoint includes args |
| Guided evaluation | Three runs; all three target slices exercised |
| Scoring-target isolation | Predictions exactly equal with original, perturbed and absent scoring targets |
| Calibration support | Outer-ACS perturbation leaves fitted coefficients exactly equal |
| Formatting/static checks | Python parsing, CSV columns and git diff whitespace pass |

The 32 dataset combinations cover file-KMB present/absent, normalization enabled/disabled, kmb/k1 normalization mode, optional K23 sqrt(2) scaling and spatial crop/pad. They do not test every possible legacy array layout. HDF5 tests use a numeric MATLAB-style compound-value/axis-order fixture, not an actual acquired clinical file.

The guided smoke run uses tiny random complex arrays, 2 coils, base channel 8 and T=2. These are software checks, not scientific performance measurements. Only the default guided evaluation branch is integration-tested; optional late/fusion branches and CUDA are not certified.

## Not verified

- Actual patient identity/splits or any clinical data.
- Historical raw-data conversion, slice grouping or CAIPI/mask settings.
- Published checkpoints, CUDA execution or paper PSNR/SSIM.
- SG/SPSG/SENSE/RAKI/KDM/ROGER/FM source implementations.
- Stage-U, accelerated configurations or MB4.
- The supplied GitHub Actions workflow on the owner's remote repository.
- A remote upload or replacement: no GitHub content was changed by this work.

See VALIDATION.json for the machine-readable run summary. Re-run real-data and baseline checks before promoting this candidate.
