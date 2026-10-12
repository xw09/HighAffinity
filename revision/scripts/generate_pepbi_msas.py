#!/usr/bin/env python3
"""Generate one reusable unpaired MMseqs2 MSA per unique PEPBI receptor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from boltz.data.msa.mmseqs2 import run_mmseqs2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUESTS = (
    ROOT / "revision/external/pepbi/inference/receptor_msa_requests.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--host-url", default="https://api.colabfold.com")
    parser.add_argument("--max-seqs", type=int, default=8192)
    return parser.parse_args()


def valid_msa(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[0] == "key,sequence" and len(lines) >= 2
    except (OSError, IndexError):
        return False


def main() -> None:
    args = parse_args()
    requests_path = args.requests.expanduser().resolve()
    with requests_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pending = [row for row in rows if not valid_msa(Path(row["msa_path"]))]
    if not pending:
        print(f"All {len(rows)} receptor MSAs already exist; nothing to do.")
        return

    digest = hashlib.sha256(
        "\n".join(row["Protein_Sequence"] for row in pending).encode("utf-8")
    ).hexdigest()[:12]
    work_dir = requests_path.parent / "mmseqs_cache" / f"unpaired_{digest}"
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Requesting {len(pending)} unpaired receptor MSAs in one batch.")
    alignments = run_mmseqs2(
        [row["Protein_Sequence"] for row in pending],
        work_dir,
        use_env=True,
        use_pairing=False,
        host_url=args.host_url,
        pairing_strategy="greedy",
    )
    if len(alignments) != len(pending):
        raise RuntimeError(
            f"MMseqs2 returned {len(alignments)} MSAs for {len(pending)} requests"
        )

    written = 0
    for row, alignment in zip(pending, alignments):
        lines = alignment.strip().splitlines()
        sequences = lines[1::2][: args.max_seqs]
        if not sequences:
            raise RuntimeError(f"Empty MSA for {row['receptor_msa_id']}")
        output = Path(row["msa_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        csv_lines = ["key,sequence"] + [f"-1,{seq}" for seq in sequences]
        output.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        written += 1

    invalid = [row["receptor_msa_id"] for row in rows if not valid_msa(Path(row["msa_path"]))]
    if invalid:
        raise RuntimeError(f"Invalid or missing MSAs after generation: {invalid[:10]}")
    marker = {
        "requested": len(pending),
        "written": written,
        "total": len(rows),
        "host_url": args.host_url,
        "max_seqs": args.max_seqs,
    }
    (requests_path.parent / "MSA_COMPLETE.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
