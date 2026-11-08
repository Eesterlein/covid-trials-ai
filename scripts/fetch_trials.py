"""Download every COVID-19 study record from the ClinicalTrials.gov v2 API.

Saves one JSON file per page (up to 1000 studies) in data/raw/ so the
download can be resumed and re-run without hitting the API again.
"""
import json
import pathlib
import time

import requests

API = "https://clinicaltrials.gov/api/v2/studies"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

params = {"query.cond": "COVID-19", "pageSize": 1000, "countTotal": "true", "format": "json"}
page, total, token = 0, 0, None

while True:
    if token:
        params["pageToken"] = token
    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    studies = data.get("studies", [])
    (OUT / f"page_{page:03d}.json").write_text(json.dumps(studies))
    total += len(studies)
    print(f"page {page}: {len(studies)} studies (running total {total} of {data.get('totalCount', '?') if page == 0 else '...'})")
    token = data.get("nextPageToken")
    if not token:
        break
    page += 1
    time.sleep(0.5)  # be polite to the API

print(f"done: {total} studies saved to {OUT}")
