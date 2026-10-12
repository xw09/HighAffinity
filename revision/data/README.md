# Revision data artifacts

Run the stage-2 audit from the repository root:

```bash
python revision/scripts/build_final_manifest.py
```

Generated files:

- `final_manifest_v2.csv`: unified trainval/test manifest with original and cleaned labels.
- `label_audit.csv`: all experimental-affinity records and parsed label decisions.
- `censored_label_audit.csv`: the 187 records without exact numerical regression labels.
- `dataset_changes.csv`: records relabeled or excluded from classification.
- `negative_type_audit.csv`: provenance audit for all originally negative records.
- `dataset_statistics.csv`: split-level counts before and after cleaning.
- `cleaning_decision_summary.csv`: counts for every cleaning decision by split.
- `excluded_classification_ids.txt`: 104 ambiguous/unknown records excluded from classification.
- `relabeled_ids.txt`: three approximate-value records relabeled from negative to positive.
- `data_quality_summary.json`: machine-readable summary and label-status counts.

The original frozen CSV files are never overwritten. `BSA` is intentionally empty at this stage and will be populated during the BSA provenance/sensitivity analysis.

## Fixed receptor-cluster split

Run:

```bash
python revision/scripts/build_cluster_split.py
```

The generated `splits/` directory contains the clean train, validation, and test ID lists; regression-only ID lists; the full split manifest; validation receptor clusters; split statistics; overlap audit; split strategy; and SHA256 checksums. The original test membership is preserved in `test_ids_all_original.txt`; ambiguous test labels are listed separately in `excluded_test_ids.txt`.
