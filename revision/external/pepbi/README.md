# PEPBI external-validation preparation

Primary source: Torres, Pantazes, and Kieslich, PEPBI dataset, Dryad DOI
`10.5061/dryad.wstqjq2wk` (CC0-1.0). Associated paper DOI:
`10.1038/s41597-025-05754-7`.

The official workbook has 329 structural entries and 181 unique values of
`PEPBI Complex Name`. A few named complexes contain multiple temperatures or
replicate affinity measurements, so exact-observation deduplication yields 188
thermodynamic observations. Multiple crystallographic copies must not be
counted as independent observations; uncertainty estimates should be clustered
by `PEPBI Complex Name` (or, more conservatively, by `Binding Group`).

## One manual download currently required

Dryad's browser page exposes the public file, but its command-line file API
returned HTTP 401/403 on 2026-09-17. Download only `PEPBI.xlsx` (about 218 KB)
from the DOI page and save it as:

`revision/external/pepbi/raw/PEPBI.xlsx`

The expected official SHA-256 is:

`0f3ae5e9021b5c0e925114c5b53a0cb441de90adc42134e5144d4ac9c4826a24`

Then run:

```bash
python revision/scripts/prepare_pepbi_external.py
```

The script performs the following fixed operations:

1. reads the official two-header-row workbook;
2. validates canonical protein and peptide sequences;
3. converts `KD (M)` to the project label `log10(KD / micromolar)` and checks
   it against thermodynamic dG where available;
4. retains all valid structural rows and creates a deduplicated experimental-
   complex table;
5. audits exact PDB, receptor, peptide, and receptor-peptide-pair overlaps;
6. calculates maximum internal receptor similarity using normalized global
   Levenshtein identity and applies the predeclared 70% cutoff;
7. writes receptor-cold and strict cohorts, plus exact-sequence-pair-disjoint
   and PDB-ID-disjoint cohorts for explicitly labelled sensitivity analyses.

The official workbook contains one internally inconsistent source row,
`ttslydif_s3_1`: its stored `KD (M)` is `1.000016`, whereas its reported ΔG
corresponds to approximately `1.36e-5 M`. The cleaner preserves the row in the
structural audit, flags it through `thermodynamic_consistency_valid=False`, and
excludes it from regression cohorts rather than guessing a replacement value.

Use `pepbi_external_strict_model_input.csv` as the primary external cohort only
if its retained sample size is adequate. With the current HighAffinity split,
the strict and receptor-cold cohorts contain zero observations because all
PEPBI entries share either a PDB source or a >=70% identical receptor with the
internal data. The pair-disjoint and PDB-disjoint outputs can still be reported
as sensitivity analyses, but must not be called receptor-cold independent
validation. Do not tune a threshold, checkpoint, or hyperparameter on any
PEPBI cohort. PEPBI is primarily a regression benchmark; do not present it as
an independent binder/non-binder ROC benchmark.
