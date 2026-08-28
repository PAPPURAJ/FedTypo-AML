# Generated manuscript artifacts

This directory contains the revised manuscript PDF, six vector figures, and
generated LaTeX macros built from the four validated result roots.

Regenerate them with:

```bash
python scripts/make_submission_assets.py \
  --ibm results/tifs_revision_v1_ibm_account_hash \
  --samld results/tifs_revision_v1_samld_account_hash \
  --ibm-secondary results/tifs_revision_partition_sensitivity_ibm_typology_skew \
  --samld-secondary results/tifs_revision_partition_sensitivity_samld_typology_skew \
  --figdir reproduced/figures \
  --tex reproduced/results_auto.tex
```

The builder validates the roots before writing assets. `results_auto.tex` is
generated and must not be edited manually.
