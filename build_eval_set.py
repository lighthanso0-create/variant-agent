"""
build_eval_set.py — sample real, labeled ClinVar variants from the
actual held-out TEST split (never used in training or in the
early-stopping validation) to build a genuine evaluation set for
evaluate.py, replacing data/sample_variants.csv's synthetic
placeholder data.

Usage:
    python3 build_eval_set.py
    python3 evaluate.py data/real_eval_variants.csv
"""

import random
import numpy as np
import pandas as pd

DATA_DIR = "/workspace/signals_seed42_zscore_trainonly_0904"
TISSUES = ["liver", "heart", "brain"]
N_PER_TISSUE_PATHO = 15
N_PER_TISSUE_BENIGN = 15
SEED = 123  # different from the training seed on purpose — no overlap with training decisions

rng = random.Random(SEED)
rows = []

for tissue in TISSUES:
    d = np.load(f"{DATA_DIR}/signals_{tissue}_test_zscore_trainonly.npz", allow_pickle=True)
    n = len(d["chrom"])
    idxs = rng.sample(range(n), min(N_PER_TISSUE_PATHO, n))
    for i in idxs:
        rows.append({
            "chrom": str(d["chrom"][i]), "pos": int(d["pos"][i]),
            "ref": str(d["ref"][i]), "alt": str(d["alt"][i]),
            "gene": "unknown", "tissue": tissue, "true_label": "pathogenic",
        })
    del d

    d = np.load(f"{DATA_DIR}/signals_benign_{tissue}_test_zscore_trainonly.npz", allow_pickle=True)
    n = len(d["chrom"])
    idxs = rng.sample(range(n), min(N_PER_TISSUE_BENIGN, n))
    for i in idxs:
        rows.append({
            "chrom": str(d["chrom"][i]), "pos": int(d["pos"][i]),
            "ref": str(d["ref"][i]), "alt": str(d["alt"][i]),
            "gene": "unknown", "tissue": tissue, "true_label": "benign",
        })
    del d

df = pd.DataFrame(rows)
df.to_csv("data/real_eval_variants.csv", index=False)
print(f"wrote {len(df)} held-out real test variants to data/real_eval_variants.csv")
print(df["true_label"].value_counts())
print(df["tissue"].value_counts())
