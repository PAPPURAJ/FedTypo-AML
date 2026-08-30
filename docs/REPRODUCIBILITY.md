# Reproducibility record

## Provenance chain

The four primary and partition-sensitivity roots identify the exact executed
source by SHA-256:

```text
201bf0fbb7a4f4082ffd5563dc2773d6a752ff0cf9e23351da09be86fc3b587d
```

That byte-exact snapshot is preserved as
`experiments/run_submission_executed_201bf0f.py`. During asset validation, a
bookkeeping defect was found in which near-zero paired differences could be
counted simultaneously as ties and wins/losses. The audited postprocessor
`scripts/fix_inference_tie_counts.py` (SHA-256
`dd6b421251231ab0fac66c49e0723f37371afd977df9912ec8cd91f42192d496`)
changed only mutually exclusive `wins`, `ties`, and `losses` counts in three
top-level inference CSVs. Each root records before/after hashes and unchanged
fields in `postprocess_inference_counts.json`. Model outputs, seed metrics,
p-values, confidence intervals, and effect sizes were unchanged.

`experiments/run_submission.py` is the current source with the tie-count rule
fixed at generation time. It also emits shared-trajectory registry-beta
metrics. The fifth root,
`tifs_registry_beta_rtx3060_v1_samld_account_hash`, records current-source
SHA-256
`6c63628be2b7e17e827fad7482d70006d43856e999fff5f87fda1c5596bf3981`.
At the preselected beta 0.15, all six metric families across both conditions
and ten seeds are byte-identical to the original primary SAML-D FedTypo files
(120 checks). The release validator checks both provenance paths.

## Reported environment

| Component | Version or value |
|---|---|
| Python | 3.13.5 (Anaconda build) |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| NumPy | 2.5.1 |
| pandas | 3.0.1 |
| scikit-learn | 1.7.2 |
| SciPy | 1.18.0 |
| NetworkX | 3.5 |
| GPU | NVIDIA GeForce RTX 3060 |

Machine-readable records, input filenames/sizes/hashes, partition mode,
optimizer policy, supported windows, seeds, methods, and configuration are in
each result root's `environment.json`.

## Experimental matrix

| Dimension | Primary evaluation | Partition sensitivity | Registry-beta sensitivity |
|---|---|---|---|
| Datasets | IBM AMLworld HI-Small; SAML-D | IBM AMLworld HI-Small; SAML-D | SAML-D |
| Partition | `account_hash` | `typology_skew` | `account_hash` |
| Status | Primary, label-independent | Secondary, transductive and outcome-conditioned | Post hoc shared-trajectory re-scoring |
| Conditions | Natural control; controlled drift | Natural control; controlled drift | Same |
| Seeds | 42–51 | 42–51 | 42–51 |
| Methods | 7 comparators/methods + 6 component ablations | FedAvg, FedTypo-NoReg, FedTypo | FedTypo at six beta values |
| Optimizer policy | One local epoch, Adam reset per window for every method | Same | Same trained trajectory per seed/condition |

IBM has 18 natural windows. The label-independent support rule retains windows
0–9, uses window 0 as warm-up, evaluates 1–9, and excludes 1,108 transactions
from the incomplete terminal tail. SAML-D retains all 19 natural windows,
warms up on window 0, and evaluates 1–18.

The primary metric concatenates all evaluated predictions within each client,
computes one AUPRC per client, and macro-averages the five clients. P@50 is a
secondary client/window alert-budget metric. Paired two-sided Wilcoxon tests
use seeds as inferential units; Holm correction is applied within each
dataset-condition family.

## Reproduction workflow

1. Acquire and verify dataset files listed in `data/README.md`.
2. Install the pinned dependencies.
3. Run `python scripts/validate_release.py` on the committed artifact.
4. Run a time-stratified `--fast-dev` integration check.
5. Execute the primary matrices with `--partition account_hash`.
6. Execute the reduced sensitivity matrices with `--partition typology_skew`
   and `--methods fedavg,fedtypo_noreg,fedtypo`.
7. Execute the SAML-D beta sweep with `--methods fedtypo` and
   `--registry-beta-grid 0,0.05,0.10,0.15,0.20,0.30`.
8. Regenerate all assets with the five-root command in the repository README.
9. Compare consolidated tables and figures at the seed-summary and confidence-
   interval level. Hardware/library differences need not yield byte-identical
   intermediate floating-point values.

The full run is compute-intensive and resumable at condition/seed/method
granularity. Transaction-level predictions are unnecessary for the paper and
are disabled by default.
