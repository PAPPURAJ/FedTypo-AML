# Compact revised result artifact

The committed tree contains four complete ten-seed result roots and no raw
transactions or transaction-level predictions:

```text
results/
  tifs_revision_v1_ibm_account_hash/
  tifs_revision_v1_samld_account_hash/
  tifs_revision_partition_sensitivity_ibm_typology_skew/
  tifs_revision_partition_sensitivity_samld_typology_skew/
```

The `account_hash` roots are primary and contain seven methods plus six
component ablations. The `typology_skew` roots are secondary, transductive,
outcome-conditioned sensitivities containing FedAvg, FedTypo-NoReg, and
FedTypo.

Each root contains environment/input/source provenance, raw window support,
seed and method summaries, two-sided paired inference, paired differences,
component-ablation inference, mechanism tests, and a tie-count postprocessing
audit manifest. Each `control_s<seed>` and `drift_s<seed>` directory contains
the relevant client, window, stream, budget, typology, prototype, registry,
inoculation, support, and event summaries plus `DONE_<method>` markers.

Seeds 42–51 and both conditions are present. Run
`python scripts/validate_release.py` from the repository root to verify the
complete matrix and provenance hashes.
