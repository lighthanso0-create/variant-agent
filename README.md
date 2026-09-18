# Variant Interpretation Agent

A small tool-use (function-calling) LLM agent that interprets genomic
variants: it checks a known-variant dataset first, falls back to a
pathogenicity prediction model when the variant is unseen, and flags
low-confidence cases for expert review instead of guessing.

This is built as an extension of an existing tissue-specific DNA
variant pathogenicity project (DNABERT-2 + epigenomic signals), to
add an LLM-orchestration layer on top of an already-working model
rather than starting an agent project from zero.

## Why this exists

Written as a focused 2-week project to get hands-on with LLM tool-use
and multi-step agent design/evaluation — the two concrete skills asked
about in the internship application's "LLM API 활용 수준" question.

## Architecture

```
                 ┌─────────────────────┐
   user query -> │   Claude (agent)     │
                 └─────────┬────────────┘
                           │
              ┌────────────▼─────────────┐
              │   1. lookup_clinvar       │  found? -> done, report label
              └────────────┬─────────────┘
                           │ not found
              ┌────────────▼─────────────┐   ┌───────────────────────────┐
              │ 2. get_reference_sequence │   │ 3. get_epigenomic_signal  │
              │  ALT-substituted DNA seq  │   │  H3K27ac/DNase, ±512bp    │
              └────────────┬─────────────┘   └─────────────┬─────────────┘
                           └───────────────┬────────────────┘
                              ┌────────────▼─────────────┐
                              │ 4. predict_pathogenicity  │  real Late Fusion
                              │  (model.py, DNABERT-2 +   │  model — NOT
                              │   epi signal + tissue)    │  predict.py's model
                              └────────────┬─────────────┘
                                           ▼
        evidence-cited verdict, or "needs expert
        review" if model confidence < 0.7
```

Tools 2-4 each have a REAL path and a MOCK fallback (deterministic,
clearly labeled) that kicks in automatically until the real files are
present — so the whole pipeline runs today.

**Important:** `model.py` here is built from the team's
`baseline2_latefusion.py` (`LateFusionModel` — sequence + epi_signal +
tissue_id, Macro AUPRC 0.880). It is deliberately NOT built from the
team's `predict.py`, which loads `DNABERT2Baseline` — a different,
stage1 DNA-only model with a different input signature. Don't swap
`model.py`'s contents for `predict.py`'s without re-checking this.

## Files

- `agent.py` — the tool-use loop and the 4-tool system prompt.
- `tools.py` — the four tools. Drop `extract_benign_signals.py` and a
  reference FASTA (via `REFERENCE_FASTA_PATH`) next to this file to
  switch tools 2-3 from mock to real; no other code changes needed.
- `model.py` — the real `LateFusionModel` architecture (copied from
  `baseline2_latefusion.py`) plus a `predict()` helper that loads a
  checkpoint and runs one variant through it. Point
  `MODEL_CHECKPOINT_PATH` at a real `best_model.pt` to use it for real.
- `normalize.py` — train-only z-score fit/apply (the leakage fix,
  reused here for Step 3 of `get_epigenomic_signal`).
- `evaluate.py` — runs the agent over a labeled test set and buckets
  every run into one of: `correct`, `flagged_for_review`,
  `wrong_coordinate_parsing`, `tissue_confusion`, `tool_call_omission`,
  `unsupported_claim`, `incorrect`, `no_final_answer`.
- `data/sample_variants.csv` — **synthetic placeholder data**, not real
  ClinVar records. Swap in the real labeled splits before drawing any
  conclusions from the eval output.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
python agent.py                        # single example query
python evaluate.py data/sample_variants.csv   # full evaluation run
```

## Connecting the real model and data

1. Point `KNOWN_VARIANTS_PATH` at a real capstone split file (matching
   the `chrom, pos, ref, alt, true_label, tissue` columns used here).
2. Copy `extract_benign_signals.py` next to `tools.py`; set
   `ZSCORE_STATS_PATH` to the real, latest
   `zscore_stats_trainonly.npy` (from
   `/workspace/signals_seed42_zscore_trainonly_0904` — NOT the older
   July `zscore_stats.npy`, which predates the leakage fix).
3. Set `REFERENCE_FASTA_PATH` to a local hg38 FASTA (needs `pyfaidx`:
   `pip install pyfaidx`) so `get_reference_sequence` stops mocking.
4. Set `MODEL_CHECKPOINT_PATH` to the real Late Fusion `best_model.pt`
   (baseline2, not baseline1/stage1) and `MODEL_DEVICE=cuda` if
   running on the GPU server.
5. Re-run `evaluate.py` and replace the results section below with
   the real numbers.

## Evaluation results

_Not yet run against real data/model — fill in after connecting them:_

| Metric | Value |
|---|---|
| Accuracy | — |
| Flagged for expert review | — |
| Failure modes observed (with counts) | — |

## Known limitations

- `get_epigenomic_signal` and `predict_pathogenicity` fall back to
  mocks until the real extractor/model files are present.
- The confidence threshold (0.7) for "needs expert review" is a first
  guess, not tuned against real precision/recall trade-offs.
- No retrieval/RAG component yet (e.g. surfacing ACMG classification
  guidelines as supporting evidence) — a natural next extension.
