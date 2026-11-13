"""Tools the chat agent can call, plus their JSON schemas for the Claude API.

Every result names its sources with IDs the answer cites in square brackets:
[NCT01234567] for a registry record, [PMC1234567] for a paper.
"""
import json

from app.retrieval import ROOT, db, search
from scripts.build_chunks import adverse_text, outcome_text

_trials = None


def trials():
    global _trials
    if _trials is None:
        _trials = {t["nct"]: t for t in json.loads((ROOT / "data" / "trials.json").read_text())}
    return _trials


def _sql_str(s):
    return s.replace("'", "''")


# ---------------------------------------------------------------- search_evidence

def search_evidence(query, source="any", include_eligibility=False, year_from=None, nct=None, k=8):
    where = ["retracted = false"]
    if not include_eligibility:
        where.append("section != 'Eligibility'")
    if source in ("paper", "registry"):
        where.append(f"source = '{source}'")
    if year_from:
        where.append(f"year >= {int(year_from)}")
    if nct:
        where.append(f"array_has(ncts, '{_sql_str(nct.upper())}')")
    hits = search(query, k=max(1, min(int(k), 15)), where=" AND ".join(where))
    if not hits:
        return "No matching passages. Try different wording, a broader source, or remove filters."
    out = []
    for h in hits:
        cite = h["pmcid"] or h["ncts"][0]
        linked = f" | trials: {', '.join(h['ncts'][:5])}" if h["pmcid"] else ""
        out.append(f"[{cite}] {h['doc_title']} ({h['year']}) | section: {h['section']}{linked} | license: {h['license']}\n{h['text']}")
    return "\n\n---\n\n".join(out)


# ---------------------------------------------------------------- find_trials

def find_trials(text=None, phase=None, study_type=None, has_results=None, randomized=None, status=None, limit=25):
    rows = list(trials().values())
    if text:
        terms = text.lower().split()
        rows = [t for t in rows if all(w in " ".join([t["title"], t["official_title"], " ".join(t["interventions"]), " ".join(t["conditions"])]).lower() for w in terms)]
    if phase:
        rows = [t for t in rows if phase.upper() in t["phases"]]
    if study_type:
        rows = [t for t in rows if t["study_type"] == study_type.upper()]
    if has_results is not None:
        rows = [t for t in rows if t["has_results"] == has_results]
    if randomized is not None:
        rows = [t for t in rows if (t["allocation"] == "RANDOMIZED") == randomized]
    if status:
        rows = [t for t in rows if t["status"] == status.upper()]
    rows.sort(key=lambda t: -(t["enrollment"] or 0))
    total = len(rows)
    lines = [f"{total} trials match (showing {min(total, limit)}, largest enrollment first)."]
    for t in rows[: max(1, min(int(limit), 100))]:
        lines.append(f"[{t['nct']}] {t['title']} | {', '.join(t['phases']) or t['study_type']} | {t['allocation'] or 'N/A'} | "
                     f"n={t['enrollment']} | {t['status']} | results posted: {t['has_results']} | papers: {', '.join(t['pmcids']) or 'none'}")
    return "\n".join(lines)


# ---------------------------------------------------------------- get_trial

MAX_CHARS = 30000


def get_trial(nct, outcome_filter=None):
    nct = nct.strip().upper()
    t = trials().get(nct)
    if not t:
        return f"No trial {nct} in this COVID-19 dataset."
    rows = db().open_table("details").search().where(f"nct = '{_sql_str(nct)}'").limit(1).to_list()
    rec = json.loads(rows[0]["record"]) if rows else {}
    parts = [
        f"[{nct}] {t['title']}",
        f"Official title: {t['official_title']}",
        f"{t['study_type']} | phase: {', '.join(t['phases']) or 'N/A'} | allocation: {t['allocation'] or 'N/A'} | masking: {t['masking'] or 'N/A'}",
        f"Status: {t['status']}{' (why stopped: ' + t['why_stopped'] + ')' if t['why_stopped'] else ''} | start: {t['start_date']} | enrollment: {t['enrollment']}",
        f"Sponsor: {t['sponsor']} | linked open-access papers: {', '.join(t['pmcids']) or 'none'}",
    ]
    for arm in rec.get("armsInterventionsModule", {}).get("armGroups", []):
        parts.append(f"Arm: {arm.get('label')} ({arm.get('type', '')}): {arm.get('description', '')[:300]}")
    res = rec.get("resultsSection")
    if not res:
        om = rec.get("outcomesModule", {})
        parts.append("No results posted on ClinicalTrials.gov. Planned primary outcomes: " +
                     "; ".join(o.get("measure", "") for o in om.get("primaryOutcomes", [])))
        return "\n".join(parts)
    outcomes = res.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
    if outcome_filter:
        words = outcome_filter.lower().split()
        outcomes = [o for o in outcomes if any(w in (o.get("title", "") + " " + o.get("description", "")).lower() for w in words)]
    parts.append(f"\nPosted results: {len(outcomes)} outcome measures{' matching the filter' if outcome_filter else ''}.")
    text = "\n".join(parts)
    shown = 0
    for o in outcomes:
        block = "\n\n" + outcome_text(o)
        if len(text) + len(block) > MAX_CHARS:
            break
        text += block
        shown += 1
    if shown < len(outcomes):
        text += (f"\n\n[{len(outcomes) - shown} more outcome measures not shown to save space: "
                 + "; ".join(o.get("title", "")[:60] for o in outcomes[shown:shown + 20])
                 + ". Call get_trial again with outcome_filter to see specific ones.]")
    if res.get("adverseEventsModule") and len(text) < MAX_CHARS:
        text += "\n\n" + adverse_text(res["adverseEventsModule"])[: MAX_CHARS - len(text)]
    return text


# ---------------------------------------------------------------- schemas

TOOLS = [
    {
        "name": "search_evidence",
        "description": "Semantic + keyword search over COVID-19 trial registry records (ClinicalTrials.gov, including posted results) "
                       "and full-text open-access papers reporting those trials. Returns the most relevant passages with source IDs. "
                       "Retracted papers are always excluded. Call several times with different wording for broad questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain language or clinical terms."},
                "source": {"type": "string", "enum": ["any", "paper", "registry"], "description": "Restrict to papers or registry records. Default any."},
                "include_eligibility": {"type": "boolean", "description": "Include eligibility criteria passages. Only useful for questions about who could enroll."},
                "year_from": {"type": "integer", "description": "Only passages from this year or later."},
                "nct": {"type": "string", "description": "Only passages linked to this trial ID, e.g. NCT04381936."},
                "k": {"type": "integer", "description": "Number of passages, 1-15. Default 8."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_trials",
        "description": "List and count trials by structured filters (design, phase, results posted). Use for 'how many', 'which trials', or to find the "
                       "largest randomized trials of an intervention before reading their results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Words that must all appear in the title, interventions or conditions, e.g. 'ivermectin'."},
                "phase": {"type": "string", "enum": ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]},
                "study_type": {"type": "string", "enum": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]},
                "has_results": {"type": "boolean", "description": "Only trials with results posted on ClinicalTrials.gov."},
                "randomized": {"type": "boolean"},
                "status": {"type": "string", "enum": ["COMPLETED", "TERMINATED", "WITHDRAWN", "RECRUITING", "ACTIVE_NOT_RECRUITING", "UNKNOWN", "SUSPENDED", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]},
                "limit": {"type": "integer", "description": "Max trials to list, 1-100. Default 25."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_trial",
        "description": "Full registry record for one trial: design, arms, enrollment, linked papers and, when posted, every outcome measure with "
                       "group sizes, values, confidence intervals and p-values, plus adverse events. Use these numbers for statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nct": {"type": "string", "description": "Trial ID, e.g. NCT04381936."},
                "outcome_filter": {"type": "string", "description": "Only outcome measures whose title or description contains these words, e.g. 'mortality'."},
            },
            "required": ["nct"],
            "additionalProperties": False,
        },
    },
]

HANDLERS = {"search_evidence": search_evidence, "find_trials": find_trials, "get_trial": get_trial}
