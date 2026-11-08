"""Map results-paper PMIDs to PMC IDs using the NCBI ID Converter API."""
import json, pathlib, time, requests
ROOT = pathlib.Path(__file__).resolve().parent.parent
studies = [s for f in sorted((ROOT/"data/raw").glob("page_*.json")) for s in json.loads(f.read_text())]
links = []  # (nct, pmid)
for s in studies:
    p = s["protocolSection"]; nct = p["identificationModule"]["nctId"]
    for ref in p.get("referencesModule", {}).get("references", []):
        if ref.get("pmid") and ref.get("type") == "RESULT":
            links.append((nct, ref["pmid"]))
pmids = sorted({pm for _, pm in links})
pm2pmc = {}
for i in range(0, len(pmids), 200):
    r = requests.get("https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
                     params={"ids": ",".join(pmids[i:i+200]), "format": "json", "tool": "covid-trials-ai"}, timeout=60)
    r.raise_for_status()
    for rec in r.json().get("records", []):
        if rec.get("pmcid"): pm2pmc[str(rec["pmid"])] = rec["pmcid"]
    time.sleep(0.4)
rows = [{"nct": n, "pmid": pm, "pmcid": pm2pmc.get(pm)} for n, pm in links]
(ROOT/"data/trial_papers.json").write_text(json.dumps(rows, indent=1))
print(f"result PMIDs: {len(pmids)} | in PMC: {len(pm2pmc)} | trials with a PMC paper: {len({r['nct'] for r in rows if r['pmcid']})}")
