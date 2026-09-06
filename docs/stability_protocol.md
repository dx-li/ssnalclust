# Corrected partition stability: frozen simulation protocol

This is an experimental evaluation, not a new public selector or an assertion
that stability identifies a scientifically correct number of clusters. Freeze
this protocol before any study solves; freeze the implementation before full
measurements. Keep failed, degenerate and no-selection outcomes. Do not extend
the grid or change seeds after seeing results.

## Literature and adaptation

[Haslbeck and Wulff (2020), §§4–5 and equation 6](https://pure.uva.nl/ws/files/54874658/Haslbeck_Wulff2020_Article_EstimatingTheNumberOfClustersV.pdf)
compare assignments on observations shared between perturbed samples. Their
corrected disagreement is the negative correlation of pairwise co-membership
indicators. Raw disagreement can favor extreme cluster counts. The correction
is imperfect; its motivating cluster-size argument makes a simplifying
assumption. Constant indicators make the corrected score undefined.

Here we adapt that score to convex-clustering strengths, with subsampling
without replacement instead of bootstrap sampling. This avoids duplicate-row
graph conventions. A strength does not specify a fixed cluster count, and
separate subsamples may produce different counts. No selection-consistency
result is claimed for this adaptation, graph rule, or numerical label threshold.

## Data and fixed choices

Nine independent simulated datasets use three scenarios and seeds 8101, 8102,
8103. Each has 120 observations and two features. For each scenario/seed, use
NumPy `default_rng(seed)` and save the generated arrays:

- `separated`: three balanced groups with means (-2,0), (2,0), (0,2), independent
  Gaussian coordinate noise with standard deviation 0.25;
- `overlap`: the same means and group sizes, with noise standard deviation 0.9;
- `single_gaussian`: 120 independent standard bivariate normal observations,
  with a single generating-population label for post-selection reporting.

For mixtures, generate 40 consecutive rows per group. Apply a row permutation
from the same generator after generating each dataset. Matching seeds across
scenarios intentionally pair random draws; do not treat all nine outcomes as
independent replicates. Separate dataset seeds are independent generated
replicates conditional on these fixed simulation families. No real dataset or
previously revealed Wine/Digits labels inform the choices.

Factors, in order, are (0, 0.1, 0.3, 1, 3, 10, 30). Each subsample independently
uses population feature standardization, a stable union 10-neighbor graph,
median positive retained-edge distance as bandwidth, and Gaussian edge weight
exp(-distance²/(2 bandwidth²)). Use the existing `prepare_training` helper with
all entries observed. Do not add connectivity edges. Strength equals factor
times that sample's bandwidth divided by 10. Report connected components.

Use SSNAL, tol 1e-6, max_iter 300, check_every 1, store_history False, and warm
starts along each increasing grid. Each scheduled fit must converge. Cluster
labels are transitive Euclidean centroid components with threshold 1e-4.
Optimization certificates do not establish exact partitions at this threshold.

## Tuning, frozen choice, and audit

For each dataset, draw five pairs of independently selected 96-row subsamples
without replacement. Sort row identities in each subsample. Use a single
`default_rng(100000 + dataset_seed)` for sequential draws. Use identical
subsample identities for every candidate factor.

For each pair, restrict both label arrays to their shared original row
identities. For all unordered distinct shared-row pairs, let A and B indicate
co-membership in the two partitions. Compute raw mean absolute disagreement
and corrected disagreement -corr(A,B), using contingency counts without
allocating a quadratic pair array. Record support counts and all exclusions.

A comparison is eligible only when both indicators have at least ten positive
and ten negative pairs. This minimum is a discretionary protocol guard, not a
literature-derived optimal threshold. Constant indicators have undefined
corrected disagreement, recorded as null. A nonconstant comparison below the
support minimum retains its score but remains ineligible.

A factor is eligible only if all five tuning comparisons are eligible. Choose
the smallest equal-weight mean corrected disagreement, breaking exact ties
by input order. If none are eligible, record **no selection**. Do not omit an
ineligible pair from a factor's mean. A failed/nonconverged scheduled fit makes
the dataset run failed, rather than changing the candidate grid.

Write a separate selection artifact before computing any audit fit. A second
seed bank, `default_rng(200000 + dataset_seed)`, generates five additional pairs
by the same sampling rule. Fit only the selected factor, with cold starts, and
report each comparison plus its mean only if all are eligible. Do not reselect
using audit results. With no selection, skip the audit and report that fact.

After the choice, fit the full-data seven-point path with newly computed
full-data preprocessing and graph. Report each point's cluster count,
singleton fraction, certificates, and post-selection adjusted Rand index.
Truth labels must never reach the selection function. Report a selected
singleton or all-fused full fit without replacing it. The support rule conditions selection on nontrivial partitions: this method
cannot test whether clusters exist. An all-fused Gaussian fit has ARI one
mechanically against its single generating label. Neither that value nor a
selected multi-group partition is a population significance test. A Gaussian
control is not a proof that a finite sample has no meaningful substructure.

## Execution and evidence

Run each dataset in a fresh sequential process, with a 600-second deadline and
six numerical thread environment variables set to one. Preserve source hashes,
environment versions, worker status, timings, graph details, and native peak
RSS when available. Save features, generating labels, sample identities,
preprocessing statistics, graph CSR arrays, fitted centers and edge duals,
and numerical diagnostics. Atomically replace checkpoint arrays before JSON;
an interrupted newer array archive can contain more fits than the old JSON
prefix describes. Hash finished artifacts from the parent process.

Independently audit saved labels, overlap scores, choices, and numerical
certificates. Figures must read recorded outputs. Report individual dataset
outcomes and limitations, without confidence intervals based on the dependent
observation pairs. New audit seeds assess resampling reproducibility on the
same dataset; they are not independent population validation. Three dataset
seeds per family are a small descriptive study, not a performance guarantee.

A separate `--smoke` run uses 24 rows, only separated/8101, factors (0,1,30),
one tuning pair and one audit pair, 80% subsamples, five neighbors, and minimum
pair support one. Strength divides bandwidth by five. The smoke run checks
execution and artifact plumbing and is not full scientific evidence.
