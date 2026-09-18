"""
agent.py — Tool-use agent for genomic variant interpretation.

Uses the Claude API's native tool-use (function calling) to run a
4-step pipeline for a variant:
  1. lookup_clinvar          — is this already a known/labeled record?
  2. get_reference_sequence  — if not, get the ALT-substituted sequence
  3. get_epigenomic_signal   — H3K27ac/DNase signal for the same variant
  4. predict_pathogenicity   — score it with the real Late Fusion model
Then synthesizes a short evidence-cited interpretation, flagging
low-confidence model calls for expert review instead of guessing.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."
    python agent.py
"""

import json
from typing import Any

import anthropic

from tools import lookup_clinvar, get_reference_sequence, get_epigenomic_signal, predict_pathogenicity

# Check docs.claude.com for the current model string before relying on this.
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a genomic variant interpretation assistant.
Given a variant (chromosome, position, reference allele, alternate allele,
gene, and tissue context), determine its likely clinical significance.

Always follow this exact order:
1. Call lookup_clinvar first. If found=True, report that label directly
   and stop — do not call the other tools.
2. If not found, call get_reference_sequence AND get_epigenomic_signal
   for the same chrom/pos/tissue given in the query (either order).
   Use the exact values from the query — do not guess or round them.
3. Call predict_pathogenicity, passing the sequence from step 2 and the
   h3k27ac/dnase values from get_epigenomic_signal.
4. If predict_pathogenicity's confidence is below 0.7, do NOT assert
   pathogenic or benign — say the variant "needs expert review" and
   name the confidence score.
5. Otherwise, give a short plain-language verdict and explicitly name
   which tool result(s) it's based on (e.g. "based on a model
   confidence of 0.82 and elevated H3K27ac signal in liver").

Respond with your final answer as plain text, not JSON.
"""

TOOLS = [
    {
        "name": "lookup_clinvar",
        "description": (
            "Look up a variant in the locally curated labeled dataset "
            "(derived from ClinVar). Returns the known label if the "
            "variant has already been reviewed, or found=False if not."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chrom": {"type": "string", "description": "Chromosome, e.g. 'chr1'"},
                "pos": {"type": "integer", "description": "1-based genomic position"},
                "ref": {"type": "string", "description": "Reference allele"},
                "alt": {"type": "string", "description": "Alternate allele"},
            },
            "required": ["chrom", "pos", "ref", "alt"],
        },
    },
    {
        "name": "get_reference_sequence",
        "description": (
            "Return the ALT-substituted DNA sequence around this variant "
            "(the model was trained on ALT sequence, not REF). Call this "
            "when lookup_clinvar returned found=False."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chrom": {"type": "string"},
                "pos": {"type": "integer"},
                "ref": {"type": "string"},
                "alt": {"type": "string"},
            },
            "required": ["chrom", "pos", "ref", "alt"],
        },
    },
    {
        "name": "get_epigenomic_signal",
        "description": (
            "Return normalized H3K27ac/DNase signal for the +/-512bp "
            "window around this position, for the given tissue. Call "
            "this when lookup_clinvar returned found=False."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chrom": {"type": "string"},
                "pos": {"type": "integer"},
                "tissue": {"type": "string", "enum": ["liver", "heart", "brain"]},
            },
            "required": ["chrom", "pos", "tissue"],
        },
    },
    {
        "name": "predict_pathogenicity",
        "description": (
            "Run the real fine-tuned Late Fusion model. Requires the "
            "sequence from get_reference_sequence and the h3k27ac/dnase "
            "values from get_epigenomic_signal. Returns a label "
            "('pathogenic'/'benign') and a confidence score 0-1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chrom": {"type": "string"},
                "pos": {"type": "integer"},
                "ref": {"type": "string"},
                "alt": {"type": "string"},
                "tissue": {"type": "string", "enum": ["liver", "heart", "brain"]},
                "sequence": {"type": "string"},
                "h3k27ac": {"type": "array", "items": {"type": "number"}},
                "dnase": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["chrom", "pos", "ref", "alt", "tissue", "sequence", "h3k27ac", "dnase"],
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_clinvar": lookup_clinvar,
    "get_reference_sequence": get_reference_sequence,
    "get_epigenomic_signal": get_epigenomic_signal,
    "predict_pathogenicity": predict_pathogenicity,
}


def run_agent(user_query: str, max_steps: int = 7, verbose: bool = True) -> dict[str, Any]:
    """Run the tool-use loop until Claude returns a final text answer."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_query}]
    trace: list[dict] = []

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            trace.append({"step": step, "type": "final_answer", "text": final_text})
            return {"answer": final_text, "trace": trace}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            result = (
                fn(**block.input) if fn else {"error": f"unknown tool '{block.name}'"}
            )

            trace.append(
                {
                    "step": step,
                    "type": "tool_call",
                    "tool": block.name,
                    "input": block.input,
                    "output": result,
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

        if verbose:
            called = [t["tool"] for t in trace if t["type"] == "tool_call" and t["step"] == step]
            print(f"[step {step}] called tool(s): {called}")

    return {"answer": None, "trace": trace, "error": "max_steps exceeded"}


if __name__ == "__main__":
    query = (
        "Interpret this variant: chr1:930165 G>A in gene SAMD11, "
        "tissue context liver."
    )
    result = run_agent(query)
    print("\n=== Final answer ===")
    print(result["answer"])
