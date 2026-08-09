# Generated artifacts

The vector figures and numerical LaTeX macros in this directory were built from the committed CSV results.

Regenerate them with:

```bash
python scripts/make_submission_assets.py \
  --ibm results/tifs_submission_v2_ibm \
  --samld results/tifs_submission_v2_samld \
  --figdir reproduced/figures \
  --tex reproduced/results_auto.tex
```

The command validates both result roots before writing any assets.
