"""Hybrid search over the chunks table: MedCPT vector similarity + keyword (BM25),
merged with reciprocal rank fusion. Retracted papers are excluded by default.

Usage:  python scripts/search.py "does ivermectin reduce hospitalization"
"""
import pathlib
import sys

import lancedb
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
_tok = _model = None


def embed_query(q):
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(QUERY_MODEL)
        _model = AutoModel.from_pretrained(QUERY_MODEL).eval()
    enc = _tok([q], truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        return _model(**enc).last_hidden_state[0, 0, :].numpy()


def search(q, k=10, where="retracted = false", pool=50):
    tbl = lancedb.connect(str(ROOT / "data" / "lancedb")).open_table("chunks")
    cols = ["chunk_id", "source", "ncts", "pmcid", "doc_title", "section", "year", "license", "text"]
    vec_hits = (tbl.search(embed_query(q), vector_column_name="vector").metric("dot")
                .where(where, prefilter=True).select(cols).limit(pool).to_list())
    kw_hits = tbl.search(q, query_type="fts").where(where, prefilter=True).select(cols).limit(pool).to_list()
    # reciprocal rank fusion: reward chunks that rank well in either list
    scores, rows = {}, {}
    for hits in (vec_hits, kw_hits):
        for rank, h in enumerate(hits):
            scores[h["chunk_id"]] = scores.get(h["chunk_id"], 0) + 1 / (60 + rank)
            rows[h["chunk_id"]] = h
    return [rows[c] | {"score": s} for c, s in sorted(scores.items(), key=lambda x: -x[1])[:k]]


if __name__ == "__main__":
    for h in search(" ".join(sys.argv[1:]) or "remdesivir mortality"):
        cite = h["pmcid"] or h["ncts"][0]
        print(f"[{h['score']:.4f}] {h['source']:8} {cite:12} {h['year']} | {h['doc_title'][:70]} | {h['section'][:40]}")
        print("    " + h["text"][:200].replace("\n", " ") + "\n")
