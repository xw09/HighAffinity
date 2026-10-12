# Compute-cost reporting notes

- Training wall time is reconstructed by summing the final completed progress-bar duration for each epoch in the archived Lightning logs.
- Inference wall time comes from the recorded start/completion timestamps for the 2,496-record held-out test evaluation.
- Inference uses precomputed Boltz structures and therefore measures the affinity inference stage, not de novo structure prediction.
- Memory values are contemporaneous `nvidia-smi` snapshots and must be reported as approximate observed usage, not exact peak allocation.
- Best epoch numbers are zero-based, matching checkpoint filenames.
- The table contains all 11 prespecified runs: three seeds each for M0 and M1, and seed 1 for M2--M6.
