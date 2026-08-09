# Result artifact

The committed result tree contains compact outputs from the complete experiments reported in the paper. It includes no raw transactions and no transaction-level prediction files.

## Structure

```text
results/
  tifs_submission_v2_ibm/
  tifs_submission_v2_samld/
```

Each dataset root contains:

| File | Description |
|---|---|
| `environment.json` | Input hashes, software versions, device, seeds, and protocol configuration |
| `seed_summary.csv` | One AUPRC and precision-at-budget record per condition, seed, and method |
| `method_summary.csv` | Ten-seed mean, standard deviation, and bootstrap confidence interval |
| `seed_level_tests.csv` | Paired one-sided Wilcoxon tests and Holm-adjusted values |

Each `control_s<seed>` or `drift_s<seed>` directory contains client, window, budget, typology, prototype, and inoculation summaries. Registry summaries are present when the run admitted registry entries. Drift directories also contain the injected event schedule.

## Methods

The seven evaluated methods are:

- `local_only`
- `fedavg`
- `fedprox`
- `fedproto`
- `cda_fedavg`
- `fedtypo_noreg`
- `fedtypo`

Seeds 42 through 51 are present for both control and injected-drift conditions on both datasets.
