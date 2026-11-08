"""Summarize the downloaded trials and collect linked PubMed IDs."""
import json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
studies = [s for f in sorted((ROOT/"data/raw").glob("page_*.json")) for s in json.loads(f.read_text())]
c = collections.Counter
types, phases, status, res = c(), c(), c(), 0
pmids, result_pmids = set(), set()
for s in studies:
    p = s["protocolSection"]
    d = p.get("designModule", {})
    types[d.get("studyType", "?")] += 1
    for ph in d.get("phases", ["NA"]): phases[ph] += 1
    status[p.get("statusModule", {}).get("overallStatus", "?")] += 1
    if s.get("hasResults"): res += 1
    for ref in p.get("referencesModule", {}).get("references", []):
        if ref.get("pmid"):
            pmids.add(ref["pmid"])
            if ref.get("type") == "RESULT": result_pmids.add(ref["pmid"])
print("studies:", len(studies), "| with posted results:", res)
print("types:", dict(types)); print("phases:", dict(phases)); print("status:", dict(status.most_common(6)))
print("linked PMIDs:", len(pmids), "| tagged as RESULT papers:", len(result_pmids))
(ROOT/"data/pmids.txt").write_text("\n".join(sorted(pmids)))
