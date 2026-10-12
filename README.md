# HighAffinity

Official implementation and reproducibility package for **HighAffinity:
Fine-tuning Boltz2 for Structure-based Protein–Peptide Binding Affinity
Prediction**.

This repository contains the audited dataset, frozen train/validation/test
identifiers, M0–M6 ablation configurations, training and evaluation code,
per-record ensemble predictions, statistical analyses, PEPBI external
validation materials, figure source data, and selected affinity-module
weights.

## Repository map

- `data/Database.csv`: original project database.
- `revision/originals/splits/`: source train/test tables used by the audit.
- `revision/data/final_manifest_v2.csv`: audited record-level dataset.
- `revision/data/splits/`: fixed IDs, split manifest, overlap audit, checksums.
- `revision/configs/public/`: portable YAMLs and resolved configurations for
  all 11 runs.
- `revision/checkpoints/affinity_heads/`: selected M0–M6 weights and SHA256
  manifest (Git LFS).
- `revision/predictions/ensembles/`: record-level three-seed M0/M1 predictions.
- `revision/statistics/` and `revision/tables/final_test/`: bootstrap, paired,
  subgroup, external-validation, and compute-cost results.
- `revision/external/pepbi/`: cleaning protocol, model inputs, overlap audit,
  and record-level external predictions.
- `revision/figures/source_data/`: numeric source data for revised figures.

Submitted documents, reviewer correspondence, local logs, full Lightning
checkpoints, and server-specific working directories are not included.

## Environment

Revision runs used Python 3.10.19, PyTorch 2.10.0 with CUDA 12.8, PyTorch
Lightning 2.5.0, bfloat16 mixed precision, and one NVIDIA GeForce RTX 4090
(24 GB) per run.

```bash
conda create -n highaffinity python=3.10 -y
conda activate highaffinity
pip install -e ./boltz
pip install -r revision/requirements-revision.txt
export PYTHONPATH="$PWD/boltz/src"
export BOLTZ_CACHE="$HOME/.boltz"
```

The public Boltz2 affinity checkpoint must be available as
`$BOLTZ_CACHE/boltz2_aff.ckpt`.

Selected affinity-head bundles use Git LFS. After cloning, retrieve them with:

```bash
git lfs install
git lfs pull --include="revision/checkpoints/affinity_heads/*.pt"
```

## Dataset audit and fixed split

```bash
python revision/scripts/build_final_manifest.py
python revision/scripts/build_cluster_split.py
sha256sum -c revision/data/splits/SHA256SUMS.txt
```

By default, the split builder replays the published receptor-group list;
`--reoptimize-validation-groups` is provided only for sensitivity work because
different MILP solver versions can select different equivalent optima.

The original held-out test membership is preserved. No held-out test record is
used for checkpoint selection, hyperparameter tuning, or threshold selection.
The classification threshold is fixed at 0.5.

## M0–M6 experiments

| Code | Training condition | Seeds |
|---|---|---|
| M0 | pretrained affinity module; Huber + focal | 1, 2, 3 |
| M1 | randomly initialized affinity module; Huber + focal | 1, 2, 3 |
| M2 | regression-only Huber objective | 1 |
| M3 | classification-only focal objective | 1 |
| M4 | MSE + focal | 1 |
| M5 | Huber + BCE | 1 |
| M6 | M0 without synthetic decoys in training | 1 |

All runs freeze the structure trunk. They use AdamW, a constant learning rate
of `1e-4`, microbatch size 3, four-step gradient accumulation (effective batch
size 12), up to eight epochs, and early stopping after two validation
evaluations without improvement. Checkpoint selection monitors
`val/loss_affinity`; disabled task terms do not contribute in M2 or M3.

Point `HIGH_AFFINITY_PROCESSED_DIR` to a prepared Boltz training directory
containing `processed/manifest.json`, then run one configuration:

```bash
export HIGH_AFFINITY_PROCESSED_DIR=/path/to/boltz_train_data
python revision/scripts/run_affinity_training.py \
  --config revision/configs/public/M0_seed1.yaml
```

## Restore selected checkpoints

Only the trainable affinity modules are distributed because the structure
trunk was frozen. A bundle plus the public Boltz2 affinity checkpoint
reconstructs the selected Lightning checkpoint used for inference:

```bash
python revision/scripts/restore_affinity_checkpoint.py \
  --base-checkpoint "$BOLTZ_CACHE/boltz2_aff.ckpt" \
  --affinity-head revision/checkpoints/affinity_heads/M0_seed1.affinity_head.pt \
  --output revision/restored/M0_seed1.ckpt
```

Verify bundles against `revision/checkpoints/affinity_heads/manifest.json`.

## Internal evaluation and statistics

```bash
python revision/scripts/run_affinity_evaluation.py \
  --config revision/configs/public/M0_seed1.yaml \
  --checkpoint revision/restored/M0_seed1.ckpt \
  --output-dir outputs/M0_seed1_validation \
  --evaluation-split validation \
  --test-dir "$HIGH_AFFINITY_PROCESSED_DIR" \
  --test-ids revision/data/splits/validation_ids.txt
```

Final test inference additionally requires
`--acknowledge-test-set-is-final`, preventing accidental test use during model
selection. Statistical analysis from aligned record-level predictions is:

```bash
python revision/scripts/analyze_predictions.py \
  --prediction M0=revision/predictions/ensembles/test/M0_ensemble/predictions.csv \
  --prediction M1=revision/predictions/ensembles/test/M1_ensemble/predictions.csv \
  --reference-run M0 \
  --bootstrap-replicates 2000 \
  --output-dir reproduced_statistics
```

Confidence intervals resample receptor groups; paired comparisons use the
same resampled groups for both models.

## PEPBI external evaluation

The PEPBI workbook is distributed by Dryad under CC0 at DOI
`10.5061/dryad.wstqjq2wk`. Its expected SHA256 is recorded in
`revision/external/pepbi/raw/SOURCE_EXPECTED_SHA256.txt`; the workbook is not
duplicated here. After downloading it to
`revision/external/pepbi/raw/PEPBI.xlsx`, run:

```bash
python revision/scripts/prepare_pepbi_external.py
python revision/scripts/prepare_pepbi_inference.py
python revision/scripts/set_pepbi_single_sequence.py
```

After Boltz preprocessing has produced a PEPBI directory containing
`processed/manifest.json`, evaluate without tuning on PEPBI:

```bash
python revision/scripts/run_pepbi_affinity_evaluation.py \
  --checkpoint revision/restored/M0_seed1.ckpt \
  --boltz-dir /path/to/processed_pepbi \
  --output-dir outputs/pepbi/M0_seed1
```

No receptor-cold records remain after the declared overlap filters. The
pair-disjoint and PDB-disjoint cohorts are therefore external sensitivity
analyses, not fully receptor-independent validation.

## Rebuild tables and figures

```bash
python revision/scripts/build_revision_table_templates.py \
  --statistics-dir revision/statistics/final_test \
  --output-dir reproduced_tables \
  --split-label held-out_test
python revision/scripts/make_revision_figures.py
```

Published numeric outputs and record-level predictions are retained so that
reported metrics can be checked without rerunning GPU inference.
