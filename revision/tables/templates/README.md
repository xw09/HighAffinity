# Revision table templates

Regenerate all CSV templates with:

```bash
python revision/scripts/build_revision_table_templates.py
```

The files are separated by reviewer claim so that unavailable results remain
explicitly marked `TBD` rather than being silently inferred.

- `table_s1_dataset_split.csv`: audited clean split and negative-source counts.
- `table_s2_ablation_validation.csv`: fixed-validation ablation results; M0 and
  M1 become mean ± sample SD after seeds 2-3 are evaluated.
- `table_s3_negative_subgroups.csv`: specificity by negative source.
- `table_s4_paired_statistics.csv`: paired receptor-cluster bootstrap effects.
- `table_s5_external_validation.csv`: deduplicated non-overlapping PEPBI
  regression results.
- `table_s6_compute_cost.csv`: parameter and compute accounting.

Required footnotes for the final tables:

1. ROC-AUC and average precision use continuous probabilities; the decision
   metrics use the fixed threshold 0.5.
2. Regression uses only exact numeric affinity labels and reports affinity as
   `log10(KD / micromolar)`.
3. Confidence intervals resample receptor groups, not individual rows.
4. M2 has no classification objective and M3 has no regression objective;
   those cells are `N/A`, not zero.
5. External PEPBI rows are deduplicated by experimental thermodynamic complex;
   multiple structural copies are not treated as independent samples.
