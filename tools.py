"""
tools.py — Tool implementations the agent can call.

Four tools now (added get_reference_sequence — the real Late Fusion
model needs sequence + epigenomic signal + tissue, and only the
epigenomic signal was covered before):

  1. lookup_clinvar          — known-variant lookup (local dataset)
  2. get_reference_sequence  — DNA sequence around the variant
  3. get_epigenomic_signal   — bigWig-derived signal for one variant
  4. predict_pathogenicity   — real Late Fusion model inference

Each of 2-4 has a REAL path and a MOCK fallback, so the agent runs
end-to-end today even before every real piece is wired in. Every mock
result is tagged "source": "MOCK ..." so it's never mistaken for a
real number in eval output.
"""

import csv
import hashlib
import os
from functools import lru_cache

from normalize import load_stats, apply_zscore

DATA_PATH = os.environ.get(
    "KNOWN_VARIANTS_PATH", os.path.join(os.path.dirname(__file__), "data", "sample_variants.csv")
)
ZSCORE_STATS_PATH = os.environ.get("ZSCORE_STATS_PATH", "zscore_stats.npy")
MODEL_CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "best_model.pt")
DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
TISSUE_TO_ID = {"liver": 0, "heart": 1, "brain": 2}


@lru_cache(maxsize=1)
def _load_known_variants() -> list[dict]:
    with open(DATA_PATH, newline="") as f:
        return list(csv.DictReader(f))


# --- Tool 1: ClinVar lookup -------------------------------------------------

def lookup_clinvar(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """Look up a variant in the locally curated labeled dataset
    (derived from ClinVar). If found, the record's stored sequence
    can be reused directly and get_reference_sequence isn't needed."""
    for row in _load_known_variants():
        if (
            row["chrom"] == chrom
            and int(row["pos"]) == int(pos)
            and row["ref"].upper() == ref.upper()
            and row["alt"].upper() == alt.upper()
        ):
            return {
                "found": True,
                "label": row["true_label"],
                "source": "local_clinvar_derived_dataset",
            }
    return {"found": False}


# --- Tool 2: reference sequence ---------------------------------------------

def get_reference_sequence(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """Return the DNA sequence around the variant, with the alt allele
    substituted in (the Late Fusion model was trained on the ALT
    sequence — see baseline2_latefusion.py's "sequence" field).

    REAL path: same logic as the team's actual extract_sequences.py
    (extract_alt_sequence) — pysam.FastaFile, window=512 (1025bp),
    validates the REF allele matches the reference at the center
    position before substituting ALT, rejects if >5% N bases.
    REFERENCE_FASTA_PATH must point at the real hg38.fa (still need
    its path on the server — ask the team).
    MOCK fallback: deterministic pseudo-sequence so the pipeline still
    runs end-to-end; DO NOT use mock output for real evaluation.
    """
    fasta_path = os.environ.get("REFERENCE_FASTA_PATH")
    window = 512
    if fasta_path and os.path.exists(fasta_path):
        try:
            import pysam
            fasta = pysam.FastaFile(fasta_path)
            if chrom not in fasta.references:
                raise ValueError(f"{chrom} not in fasta references")

            # pysam is 0-based; pos here is 1-based (same convention as
            # extract_sequences.py's validate_ref/extract_alt_sequence)
            actual_ref = fasta.fetch(chrom, pos - 1, pos).upper()
            if actual_ref != ref.upper():
                raise ValueError(f"ref mismatch at {chrom}:{pos}: fasta has {actual_ref}, query has {ref}")

            start = pos - 1 - window
            end = pos - 1 + window + 1
            chrom_length = fasta.get_reference_length(chrom)
            if start < 0 or end > chrom_length:
                raise ValueError("window exceeds chromosome boundary")

            seq = fasta.fetch(chrom, start, end).upper()
            if len(seq) != 2 * window + 1:
                raise ValueError("unexpected sequence length")
            if seq.count("N") / len(seq) > 0.05:
                raise ValueError("too many N bases (>5%)")

            center_idx = window
            alt_seq = seq[:center_idx] + alt.upper() + seq[center_idx + 1:]
            return {"sequence": alt_seq, "source": "real_reference_fasta"}
        except Exception as e:
            return {"sequence": None, "error": str(e), "source": "real_reference_fasta_failed"}

    seed = int(hashlib.sha256(f"{chrom}{pos}{ref}{alt}".encode()).hexdigest(), 16)
    bases = "ACGT"
    seq = "".join(bases[(seed >> (2 * i)) % 4] for i in range(2 * window + 1))
    seq = seq[:window] + alt.upper() + seq[window + 1:]
    return {
        "sequence": seq,
        "source": "MOCK — set REFERENCE_FASTA_PATH to the real hg38.fa to get real sequence",
    }


# --- Tool 3: epigenomic signal ----------------------------------------------

import numpy as np

# Exact EID lists from the team's real BigWig pipeline (extract_signals_max.py)
TISSUE_EIDS = {
    "liver": ["E066"],
    "heart": ["E065", "E083", "E095", "E104", "E105"],
    "brain": ["E067", "E068", "E069", "E070", "E071", "E072", "E073", "E074", "E081", "E082"],
}


def get_epigenomic_signal(chrom: str, pos: int, tissue: str) -> dict:
    """Return normalized H3K27ac/DNase signal for the +/-512bp window
    around this position, for the given tissue.

    REAL path: imports extract_max_signal from the actual
    extract_signals_max.py (copy/symlink it next to this file — it
    must be on the same server as the real BigWig files under
    WORKSPACE). Calls it once per mark (H3K27ac, DNase) with this
    tissue's EID list, matches training preprocessing order
    (log1p, then train-only z-score via normalize.py).
    MOCK fallback: used automatically if the real extractor or the
    zscore stats file isn't present yet.
    """
    try:
        from extract_signals_max import extract_max_signal
        eids = TISSUE_EIDS[tissue]
        raw_h3k27ac = np.array(extract_max_signal(chrom, pos, eids, "H3K27ac")[:1024])
        raw_dnase = np.array(extract_max_signal(chrom, pos, eids, "DNase")[:1024])
        # NOTE: verify this log1p step against the real pipeline before
        # trusting real output — postprocess_all.py applies log1p to
        # benign explicitly; pathogenic was log1p'd in an earlier fixed
        # step not shown here. Confirm both go through log1p before
        # z-score, same as at training time.
        raw_h3k27ac = np.log1p(raw_h3k27ac)
        raw_dnase = np.log1p(raw_dnase)
        stats = load_stats(ZSCORE_STATS_PATH)
        normalized_h3k27ac = apply_zscore(raw_h3k27ac, {"mean": stats["h3k_mean"], "std": stats["h3k_std"]})
        normalized_dnase = apply_zscore(raw_dnase, {"mean": stats["dnase_mean"], "std": stats["dnase_std"]})
        return {
            "h3k27ac": normalized_h3k27ac.tolist(),
            "dnase": normalized_dnase.tolist(),
            "source": "real_bigwig_extraction",
        }
    except (ImportError, FileNotFoundError, KeyError):
        seed = int(hashlib.sha256(f"{chrom}{pos}{tissue}".encode()).hexdigest(), 16)
        h3k27ac = [round((((seed >> i) % 200) - 100) / 50, 2) for i in range(0, 1024 * 2, 2)][:1024]
        dnase = [round((((seed >> (i + 1)) % 200) - 100) / 50, 2) for i in range(0, 1024 * 2, 2)][:1024]
        return {
            "h3k27ac": h3k27ac,
            "dnase": dnase,
            "source": "MOCK — extract_signals_max.py or zscore_stats.npy not found yet",
        }


# --- Tool 4: model inference -------------------------------------------------

def predict_pathogenicity(
    chrom: str, pos: int, ref: str, alt: str, tissue: str,
    sequence: str, h3k27ac, dnase,
) -> dict:
    """Run the REAL fine-tuned Late Fusion model (DNABERT-2 + epigenomic
    signal + tissue embedding — see model.py, extracted from the
    team's actual baseline2_latefusion.py). Requires sequence from
    get_reference_sequence and h3k27ac/dnase from get_epigenomic_signal
    — call both first.

    NOTE: predict.py (the other file you found) loads a DIFFERENT
    model — DNABERT2Baseline from stage1 (DNA-only, no epi/tissue
    input). That is not the model this function should use; model.py
    here is built from baseline2_latefusion.py's LateFusionModel
    instead, which is the actual best-performing checkpoint
    (Macro AUPRC 0.880).

    MOCK fallback: deterministic score derived from the inputs, used
    automatically until the checkpoint file is available.
    """
    tissue_id = TISSUE_TO_ID[tissue]
    try:
        from model import predict as model_predict
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(MODEL_CHECKPOINT_PATH)
        label, confidence = model_predict(
            sequence, h3k27ac, dnase, tissue_id, MODEL_CHECKPOINT_PATH, device=DEVICE
        )
        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "source": "dnabert2_late_fusion_model",
        }
    except (ImportError, FileNotFoundError, KeyError) as e:
        seed = abs(hash((chrom, pos, ref, alt, tissue)))
        confidence = (seed % 100) / 100
        label = "pathogenic" if confidence > 0.5 else "benign"
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "source": f"MOCK — model checkpoint not available yet ({e})",
        }
