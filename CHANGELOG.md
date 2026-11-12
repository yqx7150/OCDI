# Changelog

## 0.1.1rc1 — local packaging/data-interface candidate

- Restore packed loader and MB3 dataset based on supplied MB3R1.PY.
- Anchor data ignore rules; package documented entry modules.
- Require explicit HWC input, separate measurements from scoring targets.
- Export/validate central-ACS references; audit split manifests.
- Replace the independently-acquired-reference description with retrospective simulated ACS.
- Save predictions before optional scoring labels; load checkpoints strictly.
- Record args in new best checkpoints; fix zero-worker CPU loading and AMP fallback.
- Supply synthetic tests, CPU integration checks and a GitHub workflow.

Canonical HWC training tensors are checked against the supplied source. Ambiguous layouts are rejected; packed test-label inputs and references with nonzero outer data are deliberately disallowed. These interface changes require offline conversion and real-data verification. No paper metric or historical baseline protocol is certified.
