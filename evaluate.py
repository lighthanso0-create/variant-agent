"""
evaluate.py — Run the agent over a labeled test set and analyze failures.

Buckets every run into one of these outcome categories (matches the
internship posting's explicit ask for "에이전트 실패 유형 분석"):

  correct               — right label, tools used correctly
  flagged_for_review     — agent correctly deferred on a low-confidence case
  wrong_coordinate_parsing — a tool was called with chrom/pos not matching
                             the query (the agent misread the input)
  tissue_confusion       — a tool was called with the wrong tissue
  tool_call_omission     — required tool(s) skipped (e.g. jumped to
                             predict_pathogenicity without first calling
                             get_epigenomic_signal)
  unsupported_claim      — asserted a definitive label despite a
                             low-confidence model result
  incorrect               — wrong label, no other issue detected
  no_final_answer         — agent never produced a final answer

Usage:
    python evaluate.py data/sample_variants.csv
"""

import csv
import sys
from collections import Counter

from agent import run_agent

CONFIDENCE_THRESHOLD = 0.7


def classify_outcome(row: dict, trace: list[dict], final_answer: str | None) -> str:
    if final_answer is None:
        return "no_final_answer"

    answer_lower = final_answer.lower()
    tool_calls = [t for t in trace if t["type"] == "tool_call"]
    tools_used = [t["tool"] for t in tool_calls]

    # 1. Did any tool call use the wrong coordinates? (misread the query)
    for t in tool_calls:
        inp = t["input"]
        if "chrom" in inp and inp["chrom"] != row["chrom"]:
            return "wrong_coordinate_parsing"
        if "pos" in inp and int(inp["pos"]) != int(row["pos"]):
            return "wrong_coordinate_parsing"

    # 2. Did any tool call use the wrong tissue?
    for t in tool_calls:
        inp = t["input"]
        if "tissue" in inp and inp["tissue"] != row["tissue"]:
            return "tissue_confusion"

    # 3. Was the required tool sequence followed?
    clinvar_calls = [t for t in tool_calls if t["tool"] == "lookup_clinvar"]
    if not clinvar_calls:
        return "tool_call_omission"  # never checked the known-variant dataset at all
    clinvar_found = clinvar_calls[0]["output"].get("found", False)
    if not clinvar_found:
        required = {"get_reference_sequence", "get_epigenomic_signal"}
        if not required.issubset(tools_used):
            return "tool_call_omission"
        # predict_pathogenicity must come AFTER both prerequisite tools
        prereq_step = max(
            t["step"] for t in tool_calls if t["tool"] in required
        )
        predict_calls = [t for t in tool_calls if t["tool"] == "predict_pathogenicity"]
        if predict_calls and predict_calls[0]["step"] < prereq_step:
            return "tool_call_omission"

    # 4. Did the agent assert a definitive label despite a low-confidence model score?
    predict_calls = [t for t in tool_calls if t["tool"] == "predict_pathogenicity"]
    if predict_calls:
        confidence = predict_calls[-1]["output"].get("confidence")
        if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
            if "expert review" not in answer_lower:
                return "unsupported_claim"

    # 5. Baseline correctness
    if "expert review" in answer_lower:
        return "flagged_for_review"
    if row["true_label"].lower() in answer_lower:
        return "correct"
    return "incorrect"


def main(csv_path: str) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    outcomes: Counter = Counter()
    results = []

    for row in rows:
        query = (
            f"Interpret this variant: {row['chrom']}:{row['pos']} "
            f"{row['ref']}>{row['alt']} in gene {row.get('gene', 'unknown')}, "
            f"tissue context {row['tissue']}."
        )
        result = run_agent(query, verbose=False)
        outcome = classify_outcome(row, result["trace"], result["answer"])
        outcomes[outcome] += 1
        results.append({**row, "outcome": outcome, "answer": result["answer"]})
        print(f"{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']} -> {outcome}")

    print("\n=== Summary ===")
    total = sum(outcomes.values())
    for outcome, count in outcomes.most_common():
        print(f"{outcome}: {count}/{total} ({count/total:.0%})")

    out_path = "eval_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results written to {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_variants.csv"
    main(csv_path)
