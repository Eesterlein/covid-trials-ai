"""Embed every chunk with NCBI's MedCPT Article Encoder (free, runs locally).

Embeddings are written in shards to data/emb/ so an interrupted run resumes
where it stopped. Usage:  python scripts/embed.py [--limit N]
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "emb"
MODEL = "ncbi/MedCPT-Article-Encoder"
SHARD, BATCH = 2048, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only embed the first N chunks (for testing)")
    args = ap.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    chunks = [json.loads(l) for l in open(ROOT / "data" / "chunks.jsonl")]
    if args.limit:
        chunks = chunks[: args.limit]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    print(f"{len(chunks)} chunks on {device}")
    t0 = time.time()
    for start in range(0, len(chunks), SHARD):
        out = EMB / f"shard_{start // SHARD:04d}.npy"
        if out.exists() and not args.limit:
            continue
        part = chunks[start : start + SHARD]
        order = sorted(range(len(part)), key=lambda j: len(part[j]["text"]))  # less padding per batch
        vecs = []
        for i in range(0, len(part), BATCH):
            b = [part[j] for j in order[i : i + BATCH]]
            # MedCPT takes [title, text] pairs; the title carries document + section context
            pairs = [[f"{c['doc_title']} | {c['section']}", c["text"]] for c in b]
            enc = tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                vecs.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        sorted_vecs = np.concatenate(vecs)
        result = np.empty_like(sorted_vecs)
        result[order] = sorted_vecs  # back to original chunk order
        np.save(out, result.astype(np.float32))
        done = min(start + SHARD, len(chunks))
        rate = done / (time.time() - t0)
        print(f"{done}/{len(chunks)} chunks, {rate:.0f}/s, ~{(len(chunks) - done) / rate / 60:.0f} min left", flush=True)
    print("done")


if __name__ == "__main__":
    main()
