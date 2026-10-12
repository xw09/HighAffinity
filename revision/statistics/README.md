# Revision statistical analysis

`analyze_predictions.py` calculates all metrics from per-record continuous
scores. It does not select thresholds or checkpoints and does not access a
dataset unless its prediction CSV is passed explicitly.

The primary confidence interval is a receptor-group clustered bootstrap. A
bootstrap replicate samples receptor groups with replacement and carries all
rows from each sampled group together. Paired model comparisons use the same
sampled receptor groups for both models and report the raw difference
`metric(A) - metric(B)`. For MAE and RMSE, the separate `advantage_a` column
reverses the sign so that a positive value always favors model A.

Example for the completed stage-6 validation runs:

```bash
python revision/scripts/run_stage7_statistics.py
```

This launcher automatically discovers completed validation evaluations, runs
2,000 bootstrap replicates, and rebuilds the table templates. Use `--dry-run`
to inspect its resolved inputs. Run it once after M0/M1 seeds 2-3 have also
been evaluated. The equivalent explicit command is:

```bash
python revision/scripts/analyze_predictions.py \
  --prediction M0_seed1=revision/predictions/stage6_validation/M0_seed1/epoch0-valloss0.3735/predictions.csv \
  --prediction M1_seed1=revision/predictions/stage6_validation/M1_seed1/epoch1-valloss0.3781/predictions.csv \
  --prediction M2_seed1=revision/predictions/stage6_validation/M2_seed1/epoch7-valloss0.3714/predictions.csv \
  --prediction M3_seed1=revision/predictions/stage6_validation/M3_seed1/epoch0-valloss0.2427/predictions.csv \
  --prediction M4_seed1=revision/predictions/stage6_validation/M4_seed1/epoch2-valloss0.8755/predictions.csv \
  --prediction M5_seed1=revision/predictions/stage6_validation/M5_seed1/epoch1-valloss0.6136/predictions.csv \
  --prediction M6_seed1=revision/predictions/stage6_validation/M6_seed1/epoch4-valloss0.5499/predictions.csv \
  --reference-run M0_seed1 \
  --bootstrap-replicates 2000
```

Outputs:

- `per_run_metrics.csv`: micro metrics, confusion counts, and a sensitivity
  analysis excluding the largest receptor group.
- `cluster_bootstrap_ci.csv`: point estimates and 95% clustered-bootstrap CIs.
- `paired_comparisons.csv`: paired differences, CIs, and two-sided bootstrap
  sign probabilities.
- `paired_seed_summary.csv`: mean and sample SD of paired effects across
  matching random seeds (seed 1 vs seed 1, etc.).
- `target_macro_metrics.csv`: unweighted mean across receptor groups, with the
  number of groups on which each metric is defined.
- `negative_subgroup_metrics.csv`: specificity and prediction probabilities by
  negative-source type.
- `seed_summary.csv`: mean, sample SD, range, and number of seeds.

The bootstrap sign probability is descriptive evidence, not a multiplicity-
adjusted confirmatory test. The manuscript should emphasize effect sizes and
confidence intervals rather than treating `p < 0.05` as the only criterion.
