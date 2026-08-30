# Validated revised results

The primary endpoint is stream-level client-macro AUPRC. Values below are
means across ten paired seeds under the label-independent `account_hash`
partition. Full bootstrap intervals, raw and Holm-adjusted p-values, paired
differences, rank-biserial effects, P@50/budget summaries, and per-seed records
are committed in the result roots and generated manuscript.

## Primary account-hash results

| Dataset/condition | Local | FedAvg | FedProto | FedTypo-NoReg | FedTypo |
|---|---:|---:|---:|---:|---:|
| IBM control | 0.0013 | 0.0013 | 0.0012 | 0.0013 | **0.0036** |
| IBM drift | 0.0013 | 0.0013 | 0.0012 | 0.0013 | **0.0037** |
| SAML-D control | **0.0016** | 0.0013 | 0.0013 | 0.0013 | 0.0015 |
| SAML-D drift | 0.0018 | 0.0013 | 0.0017 | 0.0013 | **0.0020** |

On IBM, FedTypo wins all ten paired seeds against every baseline in both
conditions; each six-comparison Holm-adjusted value is 0.012. On SAML-D
control, no comparison is significant. Under SAML-D drift, FedTypo is higher
than FedAvg (`p_H=0.020`), FedProx (`0.012`), CDA-FedAvg (`0.020`), and
FedTypo-NoReg (`0.023`), but not Local or FedProto (both `0.387`).

The secondary P@50 endpoint does not reproduce the AUPRC ordering. Local has
the highest mean P@50 in both primary datasets and conditions relevant to the
headline comparisons (IBM control 0.0142, IBM drift 0.0143, SAML-D control
0.0336, SAML-D drift 0.0316).

## Registry-score sensitivity

The SAML-D beta sweep re-scores the same ten paired FedTypo trajectories at
`0`, `0.05`, `0.10`, `0.15`, `0.20`, and `0.30`; it does not retrain models or
change registry admission. The preselected `beta=0.15` files are byte-for-byte
identical to the primary FedTypo files for all six metric families, two
conditions, and ten seeds (120 checks).

- In control, mean AUPRC rises from 0.001345 at beta 0 to 0.001487 at beta
  0.15 (+10.6%), but the five-setting Holm-adjusted contrast is not
  significant (`p_H=0.312`). P@50 changes from 0.014156 to 0.013222
  (`p_H=0.938`).
- Under drift, mean AUPRC rises from 0.001311 at beta 0 to 0.001963 at beta
  0.15 (+49.8%, `p_H=0.039`). Every nonzero beta has `p_H=0.039`; the
  numerical peak is beta 0.10 at 0.001994, followed by a broad plateau through
  0.30. P@50 changes from 0.013689 to 0.016156 at beta 0.15 but is not
  significant (`p_H=0.750`).

The sweep shows a drift-ranking response without validating earlier
cross-silo capture or performance at the prespecified alert budget.

## Mechanism boundary

- **Registry:** IBM one- and three-window post-onset differences are exactly
  zero in every seed. Over all post-onset windows the registry recovers one
  additional target transaction across 1,731 event-positive records (one win,
  nine ties; `p_H=1.0`). SAML-D recovers zero for both variants across 5,851
  records at every horizon.
- **Component ablations:** none of 24 comparisons with FedTypo-NoReg survives
  Holm correction; the smallest adjusted value is 0.504.
- **Prototype fidelity:** IBM is weak (named-only purity/NMI/ARI about
  0.25/0.05–0.06/0.02 and negative cosine gap). SAML-D is moderate (ARI about
  0.17–0.19 and positive cosine gap).
- **CDA control:** FedAvg and the window-adapted CDA-FedAvg are identical for
  every seed, condition, dataset, and reported metric, so the control was
  inactive or behaviorally equivalent under this protocol.

These diagnostics support a conditional ranking result, not named-typology
recovery or a validated early cross-institution registry mechanism.

## Partition sensitivity

The `typology_skew` sensitivity uses full-graph Louvain communities and
typology-conditioned reassignment. It produces substantially larger apparent
effects (FedTypo AUPRC: IBM 0.0087/0.0083 control/drift; SAML-D
0.0065/0.0079). Because it is transductive and outcome-conditioned, it is not
used for primary inference and must not be interpreted as natural bank
boundaries.
