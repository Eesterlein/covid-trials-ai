"""For each results paper in PMC, check the public PMC Open Access S3 bucket
and download the latest version's full-text XML and metadata JSON."""
import json, pathlib, re, time, requests
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pmc"; OUT.mkdir(parents=True, exist_ok=True)
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"

def get(url, **kw):
    for attempt in range(5):
        try:
            return requests.get(url, timeout=60, **kw)
        except requests.exceptions.ConnectionError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {url}")
rows = json.loads((ROOT/"data/trial_papers.json").read_text())
pmcids = sorted({r["pmcid"] for r in rows if r["pmcid"]})

def fetch(pmcid):
    if (OUT/f"{pmcid}.xml").exists(): return pmcid, "cached"
    keys = re.findall(r"<Key>([^<]*)</Key>", get(BUCKET, params={"list-type": "2", "prefix": pmcid + "."}).text)
    versions = sorted({int(k.split("/")[0].split(".")[1]) for k in keys if k.split("/")[0].split(".")[0] == pmcid})
    if not versions: return pmcid, None
    base = f"{pmcid}.{versions[-1]}"
    for ext in ("xml", "json"):
        r = get(f"{BUCKET}/{base}/{base}.{ext}")
        if r.ok: (OUT/f"{pmcid}.{ext}").write_bytes(r.content)
    return pmcid, base

with ThreadPoolExecutor(4) as ex:
    results = dict(ex.map(fetch, pmcids))
found = {k: v for k, v in results.items() if v}
for r in rows: r["oa"] = bool(r["pmcid"] and results.get(r["pmcid"]))
(ROOT/"data/trial_papers.json").write_text(json.dumps(rows, indent=1))
print(f"PMC papers: {len(pmcids)} | open access full text: {len(found)} | trials covered: {len({r['nct'] for r in rows if r['oa']})}")
