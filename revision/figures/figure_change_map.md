# Manuscript figure replacement map

The files in `main/` are direct replacements for the current manuscript
figures. The source DOCX files have **not** been modified.

| Current manuscript item | Required action | Replacement file | Main reason |
|---|---|---|---|
| Figure 1 (`image1.tif`) | Replace | `main/figure1_revised_pipeline.pdf` or `.png` | Updates audited sample counts and fixed split; explicitly distinguishes the frozen Boltz2 trunk from the trainable affinity module. |
| Figure 2 (`image2.tif`) | Replace | `main/figure2_internal_performance.pdf` or `.png` | Uses the final 2,496-row held-out test set and 516 exact regression labels; reports three-seed M0/M1 ensembles and correctly aligned baseline curves. |
| Figure 3 (`image3.tif`) | Replace | `main/figure3_ablation_and_negative_validation.pdf` or `.png` | Removes unsupported “all metrics improved” claims and instead presents ablations, real/synthetic negative subgroups, uncertainty, and paired ensemble statistics. |
| Figure 4 (`image4.tif`) | Replace | `main/figure4_pepbi_external_validation.pdf` or `.png` | Replaces the old descriptive residual comparison with independent PEPBI validation and binding-group bootstrap uncertainty. |
| Duplicate architecture image after the end matter (`image5.tif`) | Delete | None | This is the duplicated figure identified by Reviewer 1. |

## Optional SI addition

Add `supplementary/figure_s1_dataset_audit.pdf` to the Supplementary Material.
It documents the label audit, split sizes, affinity distributions, and negative
source composition. This can remain an SI figure to avoid renumbering the main
manuscript.

## Recommended insertion format

- Use PDF or SVG while editing if Word preserves the vector object correctly.
- If Word rasterizes or changes fonts, use the 600-dpi PNG.
- Keep the figure at full two-column width (approximately 7.0–7.2 inches).
- Do not reuse the old text paragraphs below Figures 2–4; those paragraphs
  contain claims contradicted by the audited results.
