"""Load chunks + embeddings into a local LanceDB database with a keyword index.

Creates data/lancedb/ with two tables:
  chunks  text, citation metadata, 768-dim MedCPT vector, full-text index
  trials  one structured row per trial (for filters, listings and stats)
"""
import glob
import json
import pathlib

import lancedb
import numpy as np
import pyarrow as pa

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
    vecs = np.concatenate([np.load(f) for f in sorted(glob.glob(str(DATA / "emb" / "shard_*.npy")))])
    assert len(vecs) == len(chunks), f"{len(vecs)} vectors for {len(chunks)} chunks: finish embed.py first"

    db = lancedb.connect(str(DATA / "lancedb"))
    table = pa.table({
        "chunk_id": [c["chunk_id"] for c in chunks],
        "source": [c["source"] for c in chunks],
        "ncts": [c["ncts"] for c in chunks],
        "pmcid": [c["pmcid"] or "" for c in chunks],
        "doc_title": [c["doc_title"] for c in chunks],
        "section": [c["section"] for c in chunks],
        "year": pa.array([c["year"] for c in chunks], pa.int32()),
        "license": [c["license"] for c in chunks],
        "retracted": [c["retracted"] for c in chunks],
        "text": [c["text"] for c in chunks],
        # one combined column for keyword search, so titles, sections and trial IDs match too
        "search_text": [f"{c['doc_title']} {c['section']} {' '.join(c['ncts'])} {c['text']}" for c in chunks],
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel(), pa.float32()), vecs.shape[1]),
    })
    tbl = db.create_table("chunks", table, mode="overwrite")
    tbl.create_fts_index("search_text", replace=True)
    for col in ("source", "pmcid", "retracted"):
        tbl.create_scalar_index(col, replace=True)

    trials = json.loads((DATA / "trials.json").read_text())
    db.create_table("trials", trials, mode="overwrite").create_scalar_index("nct", replace=True)

    # full registry record per trial (arms, outcomes, posted results) for the get_trial tool
    details = []
    for f in sorted(glob.glob(str(DATA / "raw" / "page_*.json"))):
        for s in json.load(open(f)):
            p = s["protocolSection"]
            keep = {k: p[k] for k in ("armsInterventionsModule", "outcomesModule") if k in p}
            if "resultsSection" in s:
                keep["resultsSection"] = s["resultsSection"]
            details.append({"nct": p["identificationModule"]["nctId"], "record": json.dumps(keep)})
    db.create_table("details", details, mode="overwrite").create_scalar_index("nct", replace=True)
    print(f"chunks table: {tbl.count_rows()} rows | trials table: {len(trials)} rows | details: {len(details)} rows")


if __name__ == "__main__":
    main()
