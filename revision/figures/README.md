# Revision figures

This directory contains reproducible manuscript figures generated exclusively
from the audited revision outputs.

- `main/`: four replacement main-text figures in PDF, SVG, and 600-dpi PNG.
- `supplementary/`: optional dataset-audit figure for the SI.
- `source_data/`: plotted values at the observation or summary level.
- `figure_change_map.md`: exact old-to-new replacement instructions.
- `figure_captions.md`: English captions ready for manuscript insertion.

Regenerate all files from the repository root with:

```bash
MPLCONFIGDIR=/tmp/highaffinity_mpl \
  python revision/scripts/make_revision_figures.py
```

The generator does not modify the manuscript or Supplementary Material DOCX
files.
