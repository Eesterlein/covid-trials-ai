"""Download the search database from the private Hugging Face dataset named in DATA_REPO.

Needs HF_TOKEN (a read token) as a Space secret. Skips the download if data is already present.
"""
import os
import pathlib

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

if (DATA / "lancedb").exists() and (DATA / "trials.json").exists():
    print("data already present")
else:
    repo = os.environ["DATA_REPO"]
    print(f"downloading {repo} ...", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(DATA), token=os.environ.get("HF_TOKEN"))
    print("data ready", flush=True)
