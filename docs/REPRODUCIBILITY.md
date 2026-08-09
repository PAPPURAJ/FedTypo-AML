# Reproducibility record

This record addresses the implementation, preprocessing, training, evaluation, and hardware details requested by the [IEEE TIFS guidance for learning-based submissions](https://signalprocessingsociety.org/sites/default/files/uploads/publications_resources/docs/Guidelines__deep_learning_submissions.pdf).

## Reported environment

| Component | Version or value |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| NumPy | 2.2.6 |
| pandas | 3.0.1 |
| scikit-learn | 1.7.2 |
| SciPy | 1.16.2 |
| NetworkX | 3.5 |
| GPU | NVIDIA GeForce RTX 3060 |

The machine-readable records are stored in each dataset result root as `environment.json`. Both records identify the canonical experiment source by SHA-256:

```text
f3cc7f9844588019c656449243f0f1871e9f3d699ba3a21579fc82eafedb2d9f
```

## Experimental matrix

| Dimension | Values |
|---|---|
| Datasets | IBM AMLworld HI-Small, SAML-D |
| Conditions | Control, injected drift |
| Seeds | 42--51 |
| Clients | 5 |
| Methods | Local-only, FedAvg, FedProx, FedProto, CDA-FedAvg, FedTypo-NoReg, FedTypo |
| Primary metric | Client-macro AUPRC |
| Operational metric | Precision at 50 alerts per client and window |

## Core protocol

- Source accounts remain assigned to one client.
- The first temporal window initializes client-local state and online amount statistics and is not scored.
- Evaluation follows a prequential test-then-train order.
- Drift events are retimed consistently with the transaction stream.
- FedAvg, FedProx, and CDA-FedAvg use sample-weighted aggregation.
- Immediate alert noise uses a true-positive rate of 0.60 and a false-positive rate of 0.01.
- Confirmed labels follow a log-normal delay with a median of three evaluation cycles.
- Client results are aggregated before seed-level inference.
- Paired Wilcoxon comparisons use seeds as inferential units and apply Holm correction.

## Main configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 64 |
| Local epochs | 1 |
| Learning rate | 0.002 |
| Focal-loss gamma | 2.0 |
| Maximum prototypes per client | 4 |
| Minimum confirmed positives | 8 |
| Cross-cluster damping | 0.2 |
| Registry score weight | 0.15 |
| Novelty gate | 0.3 |
| FedProto penalty | 0.1 |

Dataset-specific natural window counts and frequencies are recorded in the environment files.

## Workflow

1. Acquire and verify the dataset files listed in `data/README.md`.
2. Install the pinned Python dependencies.
3. Run `scripts/validate_release.py`.
4. Execute the fast integration run.
5. Execute the ten-seed IBM and SAML-D runs.
6. Regenerate the figures and numerical macros from the new result roots.
7. Compare `method_summary.csv`, `seed_level_tests.csv`, and the generated figures with the committed artifact.

The full experiment is compute-intensive and resumable at the condition, seed, and method level. Exact wall-clock duration depends on storage throughput, CPU memory bandwidth, and accelerator performance.

## Expected outputs

Full runs write:

```text
<output-root>/results/tifs_submission_v2_ibm/
<output-root>/results/tifs_submission_v2_samld/
```

The expected compact file hierarchy is documented in `results/README.md`. Small floating-point differences can occur across hardware and library builds; conclusions should be compared at the seed-summary and confidence-interval level rather than by requiring byte-identical intermediate values.
