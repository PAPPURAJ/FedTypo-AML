# FedTypo

Official implementation and reproducibility artifact for **FedTypo:
Prototype-Guided Federated Anti-Money-Laundering Detection Under
Heterogeneity, Concept Drift, and Delayed Labels**.

FedTypo is a cross-silo temporal graph-ranking method. Confirmed-positive
transaction embeddings form compact behavioral prototypes; prototype-set
similarity guides grouped federated aggregation, while a delayed novelty gate
controls an optional de-duplicated registry. The revised experiments support a
stream-level ranking effect but do **not** isolate the registry as a reliable
early-transfer mechanism.

The repository contains the exact executed experiment source, the corrected
current source, compact outputs from all reported runs, environment/provenance
records, and deterministic paper-asset generation. Third-party datasets and
raw transaction predictions are not redistributed.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/run_submission_executed_201bf0f.py` | Exact source used for the four reported result roots |
| `experiments/run_submission.py` | Current source with mutually exclusive inference tie counts |
| `scripts/fix_inference_tie_counts.py` | Audited postprocessor applied only to win/tie/loss counts |
| `scripts/run_experiment.py` | Command-line experiment entry point |
| `scripts/make_submission_assets.py` | Figure and LaTeX-macro generator |
| `scripts/validate_release.py` | Standard-library integrity/completeness validator |
| `results/` | Two primary account-hash and two typology-skew sensitivity roots |
| `artifacts/figures/` | Six generated vector figures |
| `artifacts/results_auto.tex` | Generated numerical LaTeX macros |
| `artifacts/manuscript.pdf` | Revised ten-page manuscript |
| `docs/REPRODUCIBILITY.md` | Protocol, environment, provenance, and workflow |
| `docs/RESULTS.md` | Evidence-constrained result summary |

## Verify the release

The release validator uses only the Python standard library:

```bash
python scripts/validate_release.py
```

It verifies the executed-source and postprocessor hashes, dataset/partition
metadata, ten seeds, both conditions, expected method matrices, required
per-run files, finite consolidated CSVs, postprocessing manifests, generated
assets, and absence of raw predictions or old result roots.

## Install

The reported GPU runs used Python 3.13.5, PyTorch 2.9.0 with CUDA 12.8, and an
NVIDIA GeForce RTX 3060. Create an isolated environment and install the pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Install a
PyTorch build appropriate for the target CPU/CUDA platform if the default
wheel is unsuitable.

## Prepare the datasets

Download the public datasets from their original distributors and arrange:

```text
data/ibm/
  HI-Small_Trans.csv
  HI-Small_Patterns.txt

data/samld/
  SAML-D.csv
```

Expected filenames, byte sizes, and SHA-256 values are recorded in each
result root's `environment.json` and summarized in `data/README.md`.

## Run experiments

The primary partition is label-independent source-account hashing:

```bash
python scripts/run_experiment.py \
  --dataset ibm --partition account_hash \
  --data-root data/ibm --output-root outputs --seeds 10

python scripts/run_experiment.py \
  --dataset samld --partition account_hash \
  --data-root data/samld --output-root outputs --seeds 10
```

The outcome-conditioned, transductive `typology_skew` partition is a secondary
sensitivity only. To reproduce its reduced three-method matrix:

```bash
python scripts/run_experiment.py \
  --dataset ibm --partition typology_skew \
  --methods fedavg,fedtypo_noreg,fedtypo \
  --run-name tifs_revision_partition_sensitivity \
  --data-root data/ibm --output-root outputs --seeds 10
```

A fast time-stratified integration run adds `--fast-dev`. Runs are resumable;
completed method directories contain `DONE_<method>` markers. Raw prediction
retention is disabled by default.

## Regenerate manuscript assets

```bash
python scripts/make_submission_assets.py \
  --ibm results/tifs_revision_v1_ibm_account_hash \
  --samld results/tifs_revision_v1_samld_account_hash \
  --ibm-secondary results/tifs_revision_partition_sensitivity_ibm_typology_skew \
  --samld-secondary results/tifs_revision_partition_sensitivity_samld_typology_skew \
  --figdir reproduced/figures \
  --tex reproduced/results_auto.tex
```

The builder validates all four roots before writing assets. See
`docs/REPRODUCIBILITY.md` for the provenance distinction between the executed
source and the tie-count-only postprocessing step.

## Interpretation boundary

The primary endpoint is stream-level client-macro AUPRC: evaluated-window
predictions are concatenated within each client before average precision is
computed and macro-averaged across clients. P@50 is secondary. FedTypo improves
primary AUPRC consistently on IBM and selectively under SAML-D drift, while
Local is strongest at P@50. Registry recovery is essentially null, no
component ablation survives Holm correction, and IBM prototype fidelity is
weak. See `docs/RESULTS.md` for exact values and limitations.

Raw records remain local, but data locality is not differential privacy,
secure aggregation, or Byzantine robustness.

## Citation and license

Citation metadata is in `CITATION.cff`. Add final journal DOI/publication
details after acceptance. Code is available under the MIT License; dataset
licenses and publication rights remain with their distributors.
