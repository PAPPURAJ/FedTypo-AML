# Expected results

The tables below provide a compact result-level check for a complete reproduction. Values are client-macro AUPRC means across ten independent seeds. Full precision-at-budget statistics, standard deviations, bootstrap intervals, and seed-level tests are available in each dataset's `method_summary.csv` and `seed_level_tests.csv`.

## IBM AMLworld HI-Small

| Method | Control | Injected drift |
|---|---:|---:|
| Local-only | 0.3056 | 0.3487 |
| FedAvg | 0.4043 | 0.4206 |
| FedProx | 0.3986 | 0.4219 |
| FedProto | 0.2698 | 0.3322 |
| CDA-FedAvg | 0.3977 | 0.4197 |
| FedTypo-NoReg | 0.4044 | 0.4233 |
| FedTypo | 0.4251 | 0.4377 |

For the paired one-sided comparison with FedAvg, FedTypo has Holm-adjusted `p = 0.005859375` in both conditions. Under injected drift, its relative AUPRC gain over FedAvg is 4.07%.

## SAML-D

| Method | Control | Injected drift |
|---|---:|---:|
| Local-only | 0.0232 | 0.0151 |
| FedAvg | 0.0453 | 0.0308 |
| FedProx | 0.0443 | 0.0308 |
| FedProto | 0.0272 | 0.0203 |
| CDA-FedAvg | 0.0453 | 0.0307 |
| FedTypo-NoReg | 0.0451 | 0.0320 |
| FedTypo | 0.0453 | 0.0301 |

SAML-D is a transfer check rather than a second confirmatory result. FedTypo does not improve over FedAvg in the injected-drift aggregate; the Holm-adjusted comparison is `p = 1.0`. This dataset-dependent behavior is part of the reported result.

## Regeneration check

After a full rerun, compare the new aggregate CSV files with:

```text
results/tifs_submission_v2_ibm/
results/tifs_submission_v2_samld/
```

Then generate the figures and compare their trends, method ordering, confidence intervals, and reported statistical conclusions with `artifacts/figures/`.
