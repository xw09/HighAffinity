import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from revision.scripts.prepare_pepbi_external import (
    GAS_CONSTANT_KCAL,
    audit_overlap,
    canonicalize,
    deduplicate_experimental_complexes,
)


class PepbiCleaningTests(unittest.TestCase):
    def setUp(self):
        temperature = 298.15
        dg_1um = GAS_CONSTANT_KCAL * temperature * math.log(1e-6)
        self.raw = pd.DataFrame(
            {
                "Binding Group": ["G1", "G1", "G2"],
                "PEPBI Complex Name": ["g1_1", "g1_2", "g2_1"],
                "PDB ID": ["1ABC", "1ABC", "2XYZ"],
                "Crystallographic Unit": ["A1B1", "A1B1", "A1B1"],
                "Unit Copy": ["C1", "C2", "X"],
                "Protein Name": ["P1", "P1", "P2"],
                "Protein Sequence": ["ACDEFGHIK", "ACDEFGHIK", "YYYYYYYYY"],
                "Peptide Name": ["pep1", "pep1", "pep2"],
                "Peptide Sequence": ["ACDEF", "ACDEF", "VVVVV"],
                "Binding Site": ["X", "X", "X"],
                "KD (M)": [1e-6, 1e-6, 1e-9],
                "Calculated ΔG (kcal/mol)": [dg_1um, dg_1um, None],
                "T (K)": [temperature, temperature, temperature],
            }
        )

    def test_affinity_conversion_and_structure_deduplication(self):
        structures = canonicalize(self.raw)
        valid, experiments = deduplicate_experimental_complexes(structures)
        self.assertAlmostEqual(structures.loc[0, "logAffinity"], 0.0)
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(experiments), 2)
        copies = experiments.set_index("binding_group")["structure_copy_count"]
        self.assertEqual(copies["G1"], 2)

    def test_internally_inconsistent_affinity_is_not_used(self):
        raw = self.raw.iloc[[0]].copy()
        raw.loc[raw.index[0], "KD (M)"] = 1.0
        structures = canonicalize(raw)
        self.assertFalse(structures.loc[structures.index[0], "affinity_valid"])
        self.assertFalse(
            structures.loc[
                structures.index[0], "thermodynamic_consistency_valid"
            ]
        )

    def test_overlap_audit_produces_strict_and_receptor_cold_flags(self):
        _, experiments = deduplicate_experimental_complexes(canonicalize(self.raw))
        internal = pd.DataFrame(
            {
                "id": [1],
                "split_clean": ["train"],
                "Protein_Sequence": ["ACDEFGHIK"],
                "Peptide_Sequence": ["ACDEF"],
                "PDB_ID": ["1ABC"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            internal.to_csv(path, index=False)
            audit = audit_overlap(experiments, path, "all", 0.70)
        first = audit[audit["binding_group"].eq("G1")].iloc[0]
        second = audit[audit["binding_group"].eq("G2")].iloc[0]
        self.assertTrue(first["exclude_strict"])
        self.assertTrue(first["exclude_receptor_cold"])
        self.assertFalse(second["exclude_strict"])
        self.assertFalse(second["exclude_receptor_cold"])
        self.assertTrue(first["exclude_pair_disjoint"])
        self.assertTrue(first["exclude_pdb_disjoint"])
        self.assertFalse(second["exclude_pair_disjoint"])
        self.assertFalse(second["exclude_pdb_disjoint"])


if __name__ == "__main__":
    unittest.main()
