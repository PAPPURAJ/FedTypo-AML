# FedTypo

Official implementation and reproducibility artifact for **FedTypo: Typology-Aware Federated Anti-Money-Laundering Detection Under Heterogeneity, Concept Drift, and Delayed Labels**.

FedTypo is a cross-silo temporal graph-learning method for anti-money-laundering detection. Confirmed-positive transaction embeddings form compact typology prototypes. Prototype-set similarity guides client grouping for federated aggregation, while delayed prototype novelty controls a de-duplicated registry for cross-institution typology transfer.

This repository contains the experiment implementation, the compact outputs from all reported runs, the exact software and hardware records, and the scripts used to regenerate the paper figures and numerical LaTeX macros. Raw transaction-level predictions and third-party datasets are not redistributed.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/run_submission.py` | Canonical IBM AMLworld and SAML-D experiment |
| `scripts/run_experiment.py` | Command-line entry point |
| `scripts/make_submission_assets.py` | Figure and LaTeX-macro generation |
| `scripts/validate_release.py` | Integrity and completeness checks |
| `results/` | Ten-seed compact results for both datasets |
| `artifacts/figures/` | Vector figures generated from the committed results |
| `artifacts/results_auto.tex` | Numerical macros generated from the committed results |
| `data/README.md` | Dataset acquisition, layout, and checksums |
| `docs/REPRODUCIBILITY.md` | Protocol, environment, and expected outputs |
| `docs/RESULTS.md` | Expected aggregate results and interpretation |

## Verify the artifact

The validation script uses only the Python standard library.

```bash
python scripts/validate_release.py
```

It verifies the canonical source hash, both environment records, the complete control and drift seed matrix, all seven methods, aggregate tables, and the absence of raw predictions.

## Install

Python 3.13.5 was used for the reported experiments. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The reported runs used PyTorch 2.9.0 with CUDA 12.8 on an NVIDIA GeForce RTX 3060. Install the PyTorch build appropriate for the target CUDA or CPU platform if the default package does not match it.

## Prepare the data

Download the datasets from their original distribution pages and arrange one of the following layouts:

```text
data/ibm/
  HI-Small_Trans.csv
  HI-Small_Patterns.txt

data/samld/
  SAML-D.csv
```

The expected file hashes and source links are listed in [`data/README.md`](data/README.md). The loader stops with an error when a required input is missing.

## Run the experiments

A fast integration run uses a reduced input subset and one seed:

```bash
python scripts/run_experiment.py --dataset ibm --data-root data/ibm --output-root outputs --fast-dev
```

The reported ten-seed runs use seeds 42 through 51:

```bash
python scripts/run_experiment.py --dataset ibm --data-root data/ibm --output-root outputs --seeds 10
python scripts/run_experiment.py --dataset samld --data-root data/samld --output-root outputs --seeds 10
```

Runs are resumable. A completed method is skipped when its output directory contains the corresponding `DONE_<method>` marker. Transaction-level predictions are disabled by default because the paper figures and tables use compact client/window summaries.

## Regenerate the paper assets

```bash
python scripts/make_submission_assets.py \
  --ibm results/tifs_submission_v2_ibm \
  --samld results/tifs_submission_v2_samld \
  --figdir reproduced/figures \
  --tex reproduced/results_auto.tex
```

This produces the architecture, main-results, budget-sensitivity, and prototype-fidelity figures from the committed CSV files.

## Reproducibility scope

The artifact is organized around the [IEEE TIFS reproducible-research guidance](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/ieee-transactions). It supports the IBM AMLworld and SAML-D control and injected-drift experiments, seven methods, ten independent seeds, client-macro metrics, budget sensitivity, prototype fidelity, and seed-level Wilcoxon tests with Holm correction. The datasets remain under their original terms and must be obtained from their publishers.

The threat model is an honest-but-curious server with protocol-compliant clients. Raw transaction records remain local. Model parameters and typology prototypes do not constitute a formal privacy guarantee.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Add the final journal DOI and publication details after acceptance.

## License

The code is available under the [MIT License](LICENSE). Dataset licenses and publication rights are separate and remain with their respective owners.
