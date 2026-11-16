"""Export showcase answers (already generated and graded in the eval) for the web page.

Reads .claude/hillclimb/covid-qa/baseline/traces and writes app/static/examples.json, so
clicking an example question on the site costs nothing.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACES = ROOT / ".claude" / "hillclimb" / "covid-qa" / "baseline" / "traces"
PICKS = ["paxlovid_epic_hr", "dexamethasone", "remdesivir_pooled_mortality", "myocarditis", "masks"]
cases = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "evals" / "cases.jsonl")}

out = []
for cid in PICKS:
    t = json.load(open(TRACES / f"{cid}_rep0.json"))
    last_tool = max(i for i, x in enumerate(t) if x["role"] in ("tool_call", "tool_result"))
    answer = "\n\n".join(x["content"] for x in t[last_tool + 1:] if x["role"] == "assistant").strip()
    steps = []
    for x in t:
        if x["role"] == "tool_call":
            steps.append({"tool": x["name"], "input": json.loads(x["content"])})
        elif x["role"] == "tool_result" and steps and "code_execution" in steps[-1]["tool"]:
            steps[-1]["output"] = x["content"][:3000]
    out.append({"id": cid, "question": cases[cid]["prompt"], "answer": answer, "steps": steps,
                "model": "claude-opus-5-5"})
(ROOT / "app" / "static" / "examples.json").write_text(json.dumps(out, indent=1))
print(f"wrote {len(out)} examples:", [(e["id"], len(e["answer"]), len(e["steps"])) for e in out])
