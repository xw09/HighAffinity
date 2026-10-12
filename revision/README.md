# TCBB revision reproducibility package

This directory contains the public artifacts for the HighAffinity major
revision. See the repository-level `README.md` for the full workflow.

Immutable inputs and fixed identifiers are under `originals/splits/` and
`data/splits/`. Portable configurations are under `configs/public/`; selected
affinity-head bundles are under `checkpoints/affinity_heads/`; scripts are
under `scripts/`; completed numeric outputs are under `statistics/`,
`tables/final_test/`, `predictions/ensembles/`, `external/pepbi/results/`, and
`figures/source_data/`.

Full Lightning checkpoints, local logs, status files, manuscript drafts, and
reviewer correspondence are excluded. Since every public run froze the
structure trunk, each distributed affinity-head bundle restores the selected
model when merged into the public Boltz2 affinity checkpoint with
`scripts/restore_affinity_checkpoint.py`.
