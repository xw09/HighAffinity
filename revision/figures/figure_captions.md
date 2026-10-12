# Revision-ready figure captions

## Figure 1

**Figure 1. Audited data split and HighAffinity workflow.** (A) The integrated
dataset contained 20,950 records, of which 20,846 were eligible for binary
classification and 5,016 had exact quantitative affinity labels. A fixed
receptor-cluster-disjoint split assigned 16,514, 1,836, and 2,496
classification-eligible records to the training, validation, and held-out test
sets, respectively; the corresponding exact-label regression subsets contained
4,050, 450, and 516 records. (B) Protein and peptide sequences are processed by
the frozen Boltz2 structural trunk, and its interface representation is passed
to the trainable affinity module. The affinity module contains 8.74 million
trainable parameters (1.697% of 515.06 million total parameters) and jointly
outputs binder probability and logAffinity using focal classification and Huber
regression objectives. Model selection used the validation set; the internal
test set and external PEPBI data were reserved for final evaluation.

## Figure 2

**Figure 2. Performance on the receptor-cluster-disjoint held-out test set.**
(A–C) Predicted versus observed logAffinity for Boltz2 zero-shot, the
randomly initialized affinity-head control (M1), and HighAffinity with the
pretrained affinity initialization (M0), respectively. Scatter plots include
the 516 test observations with exact affinity labels; the dashed line denotes
identity. (D) Receiver operating characteristic curves and (E)
precision–recall curves for all 2,496 classification-eligible test observations.
M0 and M1 probabilities and regression predictions were averaged per record
across three independently trained seeds before metric calculation. The
comparison baselines were aligned to the same audited test identifiers. For PPAP and
ProAffinity-GNN, lower predicted log10 affinity values were treated as stronger
binding scores; this monotonic score transformation does not change ROC-AUC or
average precision. logAffinity is log10(Kd/µM), so lower values indicate
stronger binding.

## Figure 3

**Figure 3. Component ablations, negative-subgroup validation, and paired model
comparison.** (A) Classification and (B) regression metrics for the prespecified
seed-1 ablations on the held-out test set. Cell colors are scaled within each
metric column and the exact metric values are printed in the cells. M2 is the
regression-only model, M3 is classification-only, M4 replaces Huber loss with
mean squared error, M5 replaces focal loss with binary cross-entropy, and M6
excludes synthetic decoys from training. These single-seed ablations are
exploratory. (C) Specificity at a fixed 0.5 threshold for synthetic decoys,
BSA-derived negatives, and experimental negatives using the three-seed M0 and
M1 prediction ensembles. Error bars show 95% receptor-cluster bootstrap
confidence intervals. (D) M0 advantage over M1 for paired ensemble metrics;
positive values favor M0, and the RMSE difference was sign-reversed so that a
lower M0 error is positive. Error bars show 95% paired receptor-cluster
bootstrap confidence intervals based on 2,000 replicates.

## Figure 4

**Figure 4. External validation on the independent PEPBI collection.** (A)
Observed versus predicted logAffinity for the primary pair-disjoint PEPBI set
(177 observations, 137 unique sequence pairs, and 31 binding groups). (B) The
more stringent PDB-disjoint subset contained 69 observations from 35 unique
sequence pairs and four binding groups. Dashed lines in A and B denote identity.
(C) Residuals (predicted minus observed logAffinity) for the pair-disjoint set;
the orange line connects medians across six equal-frequency bins and is shown
as a descriptive trend only. (D) Pearson and Spearman correlations with 95%
binding-group bootstrap confidence intervals based on 2,000 replicates. The
wide PDB-disjoint confidence intervals reflect the presence of only four
binding groups. PEPBI was not used for training or model selection.

## Supplementary Figure S1

**Figure S1. Dataset and label audit.** (A) Flow from the 20,950 integrated
records to the 20,846 classification-eligible observations and 5,016 exact
regression labels. (B) Classification and exact-regression sample sizes in the
fixed receptor-cluster-disjoint training, validation, and test sets. (C)
Observed logAffinity distributions for exact-label observations in each split.
(D) Composition of negative observations by source. Exact regression labels
exclude censored, range-only, structure-derived, and negative-only records.
