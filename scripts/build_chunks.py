"""Turn registry records and open-access papers into searchable text chunks.

Outputs
  data/chunks.jsonl   one JSON object per chunk (text + citation metadata)
  data/trials.json    one structured row per trial, used for filters and stats
"""
import collections
import glob
import json
import pathlib
import re

from lxml import etree

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAX_WORDS = 300          # MedCPT reads up to 512 tokens; 300 words stays under that
SKIP_SECTIONS = re.compile(r"supplement|funding|acknowledg|conflict|competing|author contrib|data availab|abbreviation|ethic", re.I)


def words(s):
    return len(s.split())


def split_long(text, limit):
    """Split text longer than `limit` words on line breaks, then on sentences."""
    if words(text) <= limit:
        return [text]
    units = text.split("\n") if "\n" in text else re.split(r"(?<=[.!?;])\s+", text)
    if len(units) == 1:  # no break points at all: hard split on words
        w = text.split()
        return [" ".join(w[i:i + limit]) for i in range(0, len(w), limit)]
    out, buf = [], []
    for u in units:
        for piece in split_long(u, limit):
            if buf and words(" ".join(buf + [piece])) > limit:
                out.append("\n".join(buf)); buf = []
            buf.append(piece)
    if buf: out.append("\n".join(buf))
    return out


def pack(paragraphs, limit=MAX_WORDS, header=""):
    """Group paragraphs into chunks of at most `limit` words, splitting long ones.
    `header` (e.g. a table caption) is repeated at the top of every chunk after the first."""
    budget = limit - words(header)
    out, cur = [], []
    for p in paragraphs:
        for piece in split_long(p, budget):
            if cur and words(" ".join(cur + [piece])) > budget:
                out.append("\n\n".join(cur)); cur = []
            cur.append(piece)
    if cur: out.append("\n\n".join(cur))
    return [c if i == 0 or not header else f"{header} (continued)\n{c}" for i, c in enumerate(out)]


# ---------------------------------------------------------------- registry

def fmt_measure(m, groups):
    val = m.get("value", "NA")
    if "lowerLimit" in m or "upperLimit" in m:
        val += f" ({m.get('lowerLimit', '?')} to {m.get('upperLimit', '?')})"
    elif "spread" in m:
        val += f" ± {m['spread']}"
    return f"{groups.get(m.get('groupId'), m.get('groupId'))}: {val}"


def outcome_text(om):
    groups = {g["id"]: g.get("title", g["id"]) for g in om.get("groups", [])}
    lines = [f"{om.get('type', '').title()} outcome: {om.get('title', '')}",
             f"Time frame: {om.get('timeFrame', '')}"]
    if om.get("description"): lines.append(f"Description: {om['description']}")
    lines.append(f"Measure: {om.get('paramType', '')} {('(' + om['dispersionType'] + ')') if om.get('dispersionType') else ''}, unit: {om.get('unitOfMeasure', '')}")
    for d in om.get("denoms", [])[:1]:
        lines.append("Participants analyzed: " + "; ".join(f"{groups.get(c['groupId'], c['groupId'])}: {c['value']}" for c in d.get("counts", [])))
    for cl in om.get("classes", []):
        for cat in cl.get("categories", []):
            label = " / ".join(x for x in (cl.get("title"), cat.get("title")) if x)
            vals = "; ".join(fmt_measure(m, groups) for m in cat.get("measurements", []))
            lines.append(f"{label + ': ' if label else 'Result: '}{vals}")
    for a in om.get("analyses", []):
        parts = [f"Comparison of {', '.join(groups.get(g, g) for g in a.get('groupIds', []))}"]
        if a.get("paramType"): parts.append(f"{a['paramType']} = {a.get('paramValue', '?')}")
        if a.get("ciLowerLimit"): parts.append(f"{a.get('ciPctValue', '95')}% CI {a['ciLowerLimit']} to {a.get('ciUpperLimit', '?')}")
        if a.get("pValue"): parts.append(f"p = {a['pValue']}")
        if a.get("statisticalMethod"): parts.append(f"method: {a['statisticalMethod']}")
        lines.append("Analysis: " + ", ".join(parts))
    return "\n".join(lines)


def adverse_text(ae):
    groups = {g["id"]: g for g in ae.get("eventGroups", [])}
    lines = ["Adverse events summary"]
    for g in groups.values():
        lines.append(f"{g.get('title')}: serious {g.get('seriousNumAffected', '?')}/{g.get('seriousNumAtRisk', '?')}, "
                     f"other {g.get('otherNumAffected', '?')}/{g.get('otherNumAtRisk', '?')}, deaths {g.get('deathsNumAffected', '?')}/{g.get('deathsNumAtRisk', '?')}")
    totals = collections.Counter()
    for ev in ae.get("seriousEvents", []):
        totals[ev.get("term")] += sum(int(s.get("numAffected", 0) or 0) for s in ev.get("stats", []))
    if totals:
        lines.append("Most common serious events (participants affected, all groups): " +
                     "; ".join(f"{t} ({n})" for t, n in totals.most_common(10)))
    return "\n".join(lines)


def registry_chunks(s):
    p = s["protocolSection"]
    ident, status, design = p["identificationModule"], p.get("statusModule", {}), p.get("designModule", {})
    nct = ident["nctId"]
    title = ident.get("officialTitle") or ident.get("briefTitle", "")
    arms = p.get("armsInterventionsModule", {})
    interventions = [f"{i.get('type', '')}: {i.get('name', '')}" for i in arms.get("interventions", [])]
    start = status.get("startDateStruct", {}).get("date", "")
    trial = {
        "nct": nct, "title": ident.get("briefTitle", ""), "official_title": title,
        "status": status.get("overallStatus", ""), "study_type": design.get("studyType", ""),
        "phases": design.get("phases", []), "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "start_date": start, "year": int(start[:4]) if start[:4].isdigit() else None,
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "interventions": interventions,
        "allocation": design.get("designInfo", {}).get("allocation", ""),
        "masking": design.get("designInfo", {}).get("maskingInfo", {}).get("masking", ""),
        "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "has_results": bool(s.get("hasResults")),
        "why_stopped": status.get("whyStopped", ""),
    }
    base = {"source": "registry", "ncts": [nct], "pmcid": None, "doc_title": trial["title"],
            "year": trial["year"], "license": "public domain (ClinicalTrials.gov)", "retracted": False}
    om = p.get("outcomesModule", {})
    overview = [
        f"{trial['title']} ({nct})",
        f"Official title: {title}",
        f"Type: {trial['study_type']}; phase: {', '.join(trial['phases']) or 'N/A'}; allocation: {trial['allocation'] or 'N/A'}; masking: {trial['masking'] or 'N/A'}",
        f"Status: {trial['status']}{'; why stopped: ' + trial['why_stopped'] if trial['why_stopped'] else ''}; start: {start}; enrollment: {trial['enrollment']}",
        f"Sponsor: {trial['sponsor']}",
        f"Conditions: {', '.join(trial['conditions'])}",
        f"Interventions: {'; '.join(interventions) or 'none listed'}",
        f"Primary outcomes: {'; '.join(o.get('measure', '') for o in om.get('primaryOutcomes', []))}",
        f"Summary: {p.get('descriptionModule', {}).get('briefSummary', '')}",
    ]
    chunks = [dict(base, section="Overview", text=t) for t in pack(["\n".join(overview)], 400)]
    crit = p.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    if crit:
        chunks += [dict(base, section="Eligibility", text=t) for t in pack([crit])]
    res = s.get("resultsSection", {})
    for o in res.get("outcomeMeasuresModule", {}).get("outcomeMeasures", []):
        chunks += [dict(base, section=f"Results: {o.get('title', '')[:80]}", text=t) for t in pack([outcome_text(o)], 400, header=f"{o.get('type', '').title()} outcome: {o.get('title', '')}")]
    if res.get("adverseEventsModule"):
        chunks += [dict(base, section="Adverse events", text=t) for t in pack([adverse_text(res["adverseEventsModule"])], 400, header="Adverse events summary")]
    return trial, chunks


# ---------------------------------------------------------------- papers

def text_of(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def table_text(tw):
    cap = " ".join(text_of(x) for x in tw.iter("label", "caption"))
    rows = [" | ".join(text_of(c) for c in tr.iter("th", "td")) for tr in tw.iter("tr")]
    return f"Table: {cap}\n" + "\n".join(r for r in rows if r.strip(" |"))


def paper_chunks(pmcid, ncts, meta):
    tree = etree.parse(str(DATA / "pmc" / f"{pmcid}.xml"), etree.XMLParser(recover=True, huge_tree=True))
    art = tree.getroot()
    title = text_of(art.find(".//article-meta//article-title")) if art.find(".//article-meta//article-title") is not None else meta.get("title", "")
    yr = art.find(".//article-meta/pub-date/year")
    base = {"source": "paper", "ncts": ncts, "pmcid": pmcid, "doc_title": title,
            "year": int(yr.text) if yr is not None and yr.text.isdigit() else None,
            "license": meta.get("license_code") or "unspecified", "retracted": bool(meta.get("is_retracted"))}
    chunks = []
    for ab in art.iterfind(".//article-meta/abstract"):
        if ab.get("abstract-type") in ("graphical", "teaser"): continue
        paras = [text_of(x) for x in ab.iter("p")]
        chunks += [dict(base, section="Abstract", text=t) for t in pack(paras)]
    body = art.find(".//body")
    if body is None: return chunks

    def walk(el, path):
        paras = []
        for child in el:
            tag = child.tag if isinstance(child.tag, str) else ""
            if tag == "sec":
                t = child.find("title")
                name = text_of(t).rstrip(":") if t is not None else ""
                if SKIP_SECTIONS.search(name) or child.get("sec-type") == "supplementary-material": continue
                walk(child, path + [name] if name else path)
            elif tag == "p":
                for tw in child.iter("table-wrap"):  # tables nested inside paragraphs
                    chunks.extend(dict(base, section=" > ".join(path) or "Body", text=t) for t in pack([table_text(tw)], 400, header=table_text(tw).split(chr(10))[0][:300]))
                    tw.getparent().remove(tw)
                txt = text_of(child)
                if txt: paras.append(txt)
            elif tag == "table-wrap":
                chunks.extend(dict(base, section=" > ".join(path) or "Body", text=t) for t in pack([table_text(child)], 400, header=table_text(child).split(chr(10))[0][:300]))
        if paras:
            chunks.extend(dict(base, section=" > ".join(path) or "Body", text=t) for t in pack(paras))
    walk(body, [])
    return chunks


NCT_RE = re.compile(r"NCT\d{8}")


def trial_links(pmcid, registry_links, known):
    """Decide which trials a paper reports. Registries sometimes list another trial's paper as
    their 'result' (e.g. a cepharanthine trial citing the Paxlovid EPIC-HR paper), so the
    registration ID the paper itself states (abstract, front matter, 'registered' sentences) wins."""
    art = etree.parse(str(DATA / "pmc" / f"{pmcid}.xml"), etree.XMLParser(recover=True, huge_tree=True)).getroot()
    meta = art.find(".//article-meta")
    front = text_of(meta) if meta is not None else ""
    full = text_of(art)
    reg_sentences = " ".join(s for s in re.split(r"(?<=[.;])\s+", full) if re.search(r"regist|clinicaltrials\.gov", s, re.I))
    own = set(NCT_RE.findall(front + " " + reg_sentences)) & known
    mentioned = set(NCT_RE.findall(full))
    if own:
        return own | (registry_links & mentioned)
    return (registry_links & mentioned) or registry_links


# ---------------------------------------------------------------- main

def main():
    studies = [s for f in sorted(glob.glob(str(DATA / "raw" / "page_*.json"))) for s in json.load(open(f))]
    trials, chunks = [], []
    for s in studies:
        t, c = registry_chunks(s)
        trials.append(t); chunks += c
    n_registry = len(chunks)

    linked = collections.defaultdict(set)
    for r in json.loads((DATA / "trial_papers.json").read_text()):
        if r.get("oa"): linked[r["pmcid"]].add(r["nct"])
    known = {t["nct"] for t in trials}
    paper_ncts, relinked = {}, 0
    for pmcid, reg in linked.items():
        paper_ncts[pmcid] = trial_links(pmcid, reg, known)
        relinked += paper_ncts[pmcid] != reg
    print(f"papers whose trial links changed after checking the paper's own registration IDs: {relinked} of {len(linked)}")
    for t in trials:
        t["pmcids"] = sorted(pm for pm, ns in paper_ncts.items() if t["nct"] in ns)
    failed = []
    for pmcid, ncts in sorted(paper_ncts.items()):
        try:
            meta = json.loads((DATA / "pmc" / f"{pmcid}.json").read_text())
            chunks += paper_chunks(pmcid, sorted(ncts), meta)
        except Exception as e:  # keep going; report at the end
            failed.append((pmcid, repr(e)[:100]))

    with open(DATA / "chunks.jsonl", "w") as f:
        for i, c in enumerate(chunks):
            c["chunk_id"] = i
            f.write(json.dumps(c) + "\n")
    (DATA / "trials.json").write_text(json.dumps(trials))
    wc = [words(c["text"]) for c in chunks]
    print(f"trials: {len(trials)} | chunks: {len(chunks)} (registry {n_registry}, paper {len(chunks) - n_registry})")
    print(f"words per chunk: median {sorted(wc)[len(wc) // 2]}, max {max(wc)}, total {sum(wc):,}")
    print(f"papers failed to parse: {len(failed)} {failed[:3]}")


if __name__ == "__main__":
    main()
