# Data and calibration information budget

This is an auditable retrospective protocol, not proof that every historical baseline complied. The supplied source contains MB3R1 training, not baseline implementations or original preprocessing jobs.

## Definitions

Let k_s be single-band multi-coil k-space, D_s known CAIPI modulation, A the centered ACS mask, and P_R the multiband acquisition mask.

- Measurement: y = P_R * sum_s(D_s * k_s).
- Slice-resolved calibration: c_s = A * k_s; apply D_s to c_s when the method requires the SMS frame.
- Full slice targets are retained for scoring, excluded from reconstruction except for the declared calibration samples and offline simulation.

The utility starts with **already modulated** packed K_s = D_s * k_s and exports A * K_s. It does not implement raw fastMRI conversion, slice-group selection, CAIPI modulation or historical mask-offset selection. A specific baseline may need known demodulation or another documented representation of the same ACS samples.

Slice-resolved ACS cannot generally be obtained merely by cropping the collapsed multiband signal. This protocol assumes additional single-band slice-resolved reference information, simulated retrospectively. Do not call it independently acquired calibration.

## Support and acceleration

ACS zero-based indices: start = W//2 - L//2, stop = start + L, using the half-open band [start,stop). Readout H is retained; PE is W. Record dimensions, coils, preprocessing, phase ramps, slice groups, R, mask offset, ACS size, kernels and regularization.

Every fitting patch, not just its target center, must remain within ACS. Shared information support does not require the same mathematical fitting objective across methods.

Describe separate calibration availability and overhead. Nominal in-plane R or MB*R is not a demonstrated net scan-time gain after reference overhead. If the multiband mask includes added ACS, record its actual sampling fraction as well.

## Information access

| Phase | Complete labels | Current-scan ACS | Selection |
| --- | --- | --- | --- |
| Training subjects | Supervision/interpolation allowed | According to training design | No test feedback |
| Validation subjects | Scoring/model selection allowed | Fixed validation protocol | No final-test selection |
| Offline test preparation | Simulate fixed measurement, reference and scoring target | Export declared support only | No parameter tuning |
| Test reconstruction | No unrestricted label access | Scan-specific calibration allowed within support | No test-metric tuning |
| Final scoring | Read after prediction | No extra calibration | Fixed metric protocol |

Split by subject before slice grouping and augmentation. Different slices, groups, noise variants, scans and contrasts from the same subject must not cross splits unless a separately justified design permits it. File names alone do not prove independence. The audit tool checks declared IDs, source volumes, paths and optional duplicate file bytes, not the authenticity of the ID mapping.

## Baseline audit

Equal availability does not require every method to consume a reference. Methods without reference-dependent components need not use ACS.

| Method | Check in actual code | Status |
| --- | --- | --- |
| SG | Slice-kernel sources, targets and all fitting patches restricted to SB ACS | Source not supplied |
| SPSG | Separation and leakage-suppression calibration restricted to the same support | Source not supplied |
| SENSE | Coil maps from ACS; any known demodulation; no full-target maps | Source not supplied |
| RAKI | Scan-specific fitting on permitted ACS only; no selection using test metrics | Source not supplied |
| KDM | Enumerate reference, maps, operators and initialization; restrict reference-dependent components to ACS | Source not supplied |
| ROGER | Same dependency audit; distinguish offline learning from scan-specific estimation | Source not supplied |
| FM | Train on training subjects; test conditioning and initialization use allowed inputs | Source not supplied |
| OCDI MB3R1 | Measurement-based normalization, central-ACS operator, scoring labels excluded | Synthetic checks supplied |
| Stage-U/in-plane completion | PPA reference, missing-line labels, mask and consistency | Implementation not supplied |

Record exact code version, checkpoint hash, reference generator, data paths, commands, ACS support, calibration artifacts and metric settings. Inspect coil compression/PCA, sensitivity maps, whitening, masks, scale estimation, initialization and stopping rules for hidden access to full test targets. Label use for supervised training is not evidence of test leakage.

## Invariance checks

1. Fix measurement y and ACS c_s; alter or remove scoring targets. Predictions must not change beyond predeclared numerical tolerance.
2. Fix ACS and perturb original slice values outside it; calibration coefficients must not change. Do NOT regenerate measurement y in this test.
3. Changing measurements or allowed ACS may legitimately change predictions/calibration.
4. Compare actual archived samples against the original loader and verify subject splits.

A synthetic check certifies its exercised path, not the complete historical study.

## Suggested manuscript wording after historical verification

For all retrospective comparisons, SG, SPSG, SENSE, RAKI, KDM, ROGER, FM, and OCDI were provided with the same multiband measurements and the same permitted calibration information. Where slice or in-plane reference data were required, slice-resolved references were simulated using only the central ACS band of the corresponding single-band k-space data. All reference-dependent calibration and coil-sensitivity estimation were restricted to this support; reference information outside the ACS band was not used. Methods without reference-dependent components did not require additional reference input. These references were simulated calibration data, rather than separately acquired reference scans. Full single-band test targets were used for retrospective data generation and final evaluation; general model training, hyperparameter selection and checkpoint selection used only the predefined training and validation splits. Test-time scan-specific calibration, where required, used only the permitted ACS data.

Use this wording only when it matches executed experiments. If historical references used broader support, disclose and rerun affected comparisons under the intended protocol; changing prose alone is insufficient.
