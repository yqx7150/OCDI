# Release candidate checklist

## Implemented

- Missing data modules restored; data schemas split training/measurement/ACS/scoring.
- Nonzero outer-ACS references rejected; reference fitting masks to ACS.
- Synthetic complex MAT/HDF5, mask, split and data-contract tests.
- Tiny training plus guided evaluation with GT-invariance checks.
- Package discovery includes documented modules; root-only data ignore rules.
- CPU CI workflow supplied, not yet executed on the owner's GitHub.

Read VALIDATION.md for actual checks and their limits.

## Author verification before a full paper release

- [ ] Verify actual subject/volume splits and their origin.
- [ ] Recover raw-to-packed preprocessing, slice groups, CAIPI and sampling masks.
- [ ] Confirm reference support and disclose calibration assumptions/overhead.
- [ ] Audit SG/SPSG/SENSE/RAKI/KDM/ROGER/FM and in-plane/Stage-U code.
- [ ] Compare original and restored loaders on actual archived samples.
- [ ] Hash/match paper checkpoints and full configuration metadata.
- [ ] Run real-sample and full held-out evaluations.
- [ ] If input/calibration support changes, recompute affected comparisons.
- [ ] Record exact experiment CUDA/dependency environment.
- [ ] Supply MB4 and accelerated stages if claiming those reproductions.
- [ ] Match metric normalization/gain/clipping/masks across methods.
- [ ] Choose a code license with the rights holder and respect data terms.
- [ ] Push a reviewable fix, pass CI and create a versioned release.

Do not claim all-configurations reproducibility or verified historical absence of leakage from these synthetic checks. Document the repair in history.
