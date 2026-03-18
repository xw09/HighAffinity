# export_pocket_maps.py
import json
import numpy as np
from pathlib import Path
import torch

from boltz.data.types import Manifest
from boltz.data.module.trainingv2 import load_input  # 你上面贴的那个 load_input
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.crop.affinity import AffinityCropper
from boltz.data import const

OUT_DIR = Path("/home/fuxin/lab/xw/boltz/examples/affinity_ic50/boltz_results")  # 你的 out_dir
PROC    = OUT_DIR / "processed"
MSA_DIR = PROC / "msa"      # 现有的 MSA 目录（不重要，这里只是走一遍管线）
TGT_DIR = OUT_DIR / "predictions"

def to_np(x):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)
    return x[0] if x.ndim == 2 else x  # 去 batch 维

def main():
    manifest = Manifest.load(PROC / "manifest.json")
    tok = Boltz2Tokenizer()
    feat = Boltz2Featurizer()
    crop = AffinityCropper()

    feat_dir = PROC / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for rec in manifest.records:
        out_npz = feat_dir / f"{rec.id}.npz"
        if out_npz.exists():
            continue

        # 复用预测同款的加载逻辑，保证一致性
        inp = load_input(
            record=rec,
            target_dir=TGT_DIR,
            msa_dir=MSA_DIR,
            affinity=True,
        )

        toks = tok.tokenize(inp)
        toks = crop.crop(toks, max_tokens=256, max_atoms=2048)

        # 与你预测/训练时一致的 featurizer 配置
        features = feat.process(
            toks,
            molecules={},               # 这里不需要分子库就能得到 token 映射；若你管线里强依赖可按需填
            random=np.random.default_rng(42),
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=None,
            compute_constraint_features=True,
            override_method="other",
            compute_affinity=True,
        )

        pocket_min = {
            "token_index":    to_np(features["token_index"]),
            "mol_type":       to_np(features["mol_type"]),
            "token_pad_mask": to_np(features["token_pad_mask"]),
            "asym_id":        to_np(features.get("asym_id", None)) if "asym_id" in features else None,
            "residue_index":  to_np(features.get("residue_index", None)) if "residue_index" in features else None,
        }
        pocket_min = {k: v for k, v in pocket_min.items() if v is not None}
        np.savez_compressed(out_npz, **pocket_min)
        print(f"[OK] wrote {out_npz}")

if __name__ == "__main__":
    main()
