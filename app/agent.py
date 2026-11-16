"""The chat agent: Claude + retrieval tools + Anthropic's code-execution sandbox.

ask(messages, on_event) runs one user turn to completion and returns the final
answer text, the updated message history and token usage.
"""
import json
import os
import pathlib

import anthropic
from dotenv import load_dotenv

from app.tools import HANDLERS, TOOLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("MODEL", "claude-haiku-5-5")  # chosen from the 2026-11-12 eval: 19/20 correct at ~$0.004/question
EFFORT = os.environ.get("EFFORT", "medium")
MAX_STEPS = 8  # model calls per user turn (caps cost per question)

SYSTEM = """You are COVID Trials AI, a research assistant that answers questions about COVID-19 from a database of 10,269 ClinicalTrials.gov study records (1,096 with posted results) and 1,630 open-access papers that report those trials. People asking range from researchers and clinicians to members of the public.

How to answer:
- If a question is not about COVID-19, say first that it is outside what this database covers. Any general background after that must be labeled as such and carry no citations, since nothing in the database supports it.
- Base factual claims on what your tools return, and cite each one right after the claim with its ID in square brackets, like [NCT04381936] or [PMC9006771]. Each citation must point to a passage or record you retrieved that states that specific claim; a paper that merely discusses the same trial is not a source for it. If you can't tie a claim to a source, drop the claim or label it as general background. If the database lacks good evidence, say so plainly instead of filling gaps from memory. Before saying the database lacks something, search for it.
- Weigh the evidence. Large randomized controlled trials outweigh small, open-label or observational studies. Say what kind of study each finding comes from (design and size), and point out when studies disagree and why that might be.
- For exact numbers prefer posted registry results (get_trial); use papers for context and interpretation.
- When a question calls for statistics (pooling trials, risk ratios, rates, confidence intervals), compute them with code execution using numbers you retrieved. Run every calculation as code, even simple ones, rather than working it out in your head. Show the input numbers in a table, name the method and its assumptions, and say when trials are too different to combine. Never present a single pooled estimate across different drugs or interventions, even if asked; report each trial's result separately and explain why one combined number would mislead. Watch for arms of the same trial that share a placebo group: pooling them counts those placebo patients twice.
- Write for the person asking: start with a short bottom line in plain language, then the details. Briefly define technical terms.
- This is research information, not medical advice. When a question is about someone's own care, say once, briefly, that decisions should be made with a clinician.
- Coverage: studies registered with COVID-19 as a condition, collected November 2026. Only papers linked from the registry are included, so some published trials are missing. Mention this when it limits an answer."""

SERVER_TOOLS = [{"type": "code_execution_20260521", "name": "code_execution"}]

client = anthropic.Anthropic()


def _run_tool(block):
    try:
        out = HANDLERS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": out}
    except Exception as e:  # tell Claude what went wrong instead of crashing the turn
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"Tool error: {e}", "is_error": True}


def ask(messages, on_event=lambda kind, data: None, model=None, effort=None):
    """Run one user turn. `messages` must end with the user's message; it is extended in place."""
    model, effort = model or MODEL, effort or EFFORT
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "steps": 0, "served_model": None}
    response = None
    for step in range(MAX_STEPS):
        # if a safety classifier declines, re-run on Anthropic's recommended model (Haiku has no server-side fallback)
        fallback = {} if "haiku" in model else {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        # on the last allowed step, no more tool calls: answer from what has been found so far
        # (also keeps the history valid - it never ends on an unanswered tool call)
        last = {"tool_choice": {"type": "none"}} if step == MAX_STEPS - 1 else {}
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS + SERVER_TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},  # caches system + tools + history between loop steps
            **fallback,
            **last,
        )
        u = response.usage
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
        usage["steps"] += 1
        usage["served_model"] = response.model
        messages.append({"role": "assistant", "content": response.content})

        for b in response.content:
            if b.type == "server_tool_use":
                on_event("code", b.input)
            elif b.type == "bash_code_execution_tool_result":
                c = b.content
                on_event("code_result", getattr(c, "stdout", "") or getattr(c, "stderr", "") or getattr(c, "error_code", ""))
            elif b.type == "tool_use":
                on_event("tool", {"name": b.name, "input": b.input})

        if response.stop_reason == "refusal":
            on_event("refusal", getattr(response, "stop_details", None))
            return "Sorry, I can't help with that question.", messages, usage
        if response.stop_reason == "pause_turn":
            continue  # server tool still working; send the turn back to continue
        if response.stop_reason == "tool_use":
            results = [_run_tool(b) for b in response.content if b.type == "tool_use"]
            for r in results:
                on_event("tool_result", r["content"][:300])
            messages.append({"role": "user", "content": results})
            continue
        break  # end_turn or max_tokens

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason not in ("end_turn", "stop_sequence"):
        text += f"\n\n_(Answer may be incomplete: stopped with reason `{response.stop_reason}`.)_"
    return text, messages, usage


# per-million-token prices for the cost estimate (input, output, cache read, cache write)
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20, 5.00), "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
          "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}


def cost(usage, model=MODEL):
    i, o, cr, cw = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (usage["input"] * i + usage["output"] * o + usage["cache_read"] * cr + usage["cache_write"] * cw) / 1e6
