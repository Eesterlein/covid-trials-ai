"""Hybrid search over the chunks table: MedCPT vector similarity + keyword (BM25),
merged with reciprocal rank fusion, then reranked by the MedCPT cross-encoder.
Retracted papers and eligibility criteria are excluded by default.

Try it:  python -m app.retrieval "does ivermectin reduce hospitalization"
"""
import pathlib
import sys
import threading

import lancedb
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"
_tok = _model = _rtok = _rmodel = None
_load_lock = threading.Lock()


def load_models():
    """Load the query encoder and reranker once. Safe to call from many threads:
    concurrent first-time loads otherwise corrupt each other (meta-device tensor errors)."""
    global _tok, _model, _rtok, _rmodel
    with _load_lock:
        if _model is None:
            _tok = AutoTokenizer.from_pretrained(QUERY_MODEL)
            _model = AutoModel.from_pretrained(QUERY_MODEL).eval()
        if _rmodel is None:
            _rtok = AutoTokenizer.from_pretrained(RERANK_MODEL)
            _rmodel = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).eval()


def embed_query(q):
    load_models()
    enc = _tok([q], truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        return _model(**enc).last_hidden_state[0, 0, :].numpy()


def rerank(q, rows):
    """Score each (question, passage) pair together with the MedCPT cross-encoder."""
    load_models()
    pairs = [[q, f"{r['doc_title']} | {r['section']} | {r['text']}"] for r in rows]
    enc = _rtok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = _rmodel(**enc).logits.squeeze(dim=1).tolist()
    return [r | {"score": s} for s, r in sorted(zip(logits, rows), key=lambda x: -x[0])]


DEFAULT_WHERE = "retracted = false AND section != 'Eligibility'"


_db = None


def db():
    global _db
    if _db is None:
        _db = lancedb.connect(str(ROOT / "data" / "lancedb"))
    return _db


def search(q, k=10, where=DEFAULT_WHERE, pool=50, rerank_top=30):
    tbl = db().open_table("chunks")
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
    fused = [rows[c] for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:rerank_top]]
    return rerank(q, fused)[:k]


if __name__ == "__main__":
    for h in search(" ".join(sys.argv[1:]) or "remdesivir mortality"):
        cite = h["pmcid"] or h["ncts"][0]
        print(f"[{h['score']:6.2f}] {h['source']:8} {cite:12} {h['year']} | {h['doc_title'][:70]} | {h['section'][:40]}")
        print("    " + h["text"][:200].replace("\n", " ") + "\n")
