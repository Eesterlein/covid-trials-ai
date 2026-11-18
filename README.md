# COVID Trials AI

A chat assistant that answers questions about COVID-19 from the clinical trial evidence itself:
10,269 ClinicalTrials.gov records and 1,630 open-access papers that report those trials. Every
claim in an answer links to the trial record or paper it came from, and any statistics (risk
ratios, pooled estimates, numbers needed to treat) are computed in code, with the code shown.

**Live demo:** [huggingface.co/spaces/Eesterlein/covid-trials-ai](https://huggingface.co/spaces/Eesterlein/covid-trials-ai)

Research information, not medical advice.

## What it does

Ask something like *"Did hydroxychloroquine help hospitalized COVID patients?"* The assistant:

1. searches trial records and papers (meaning-based and keyword search, then a medical reranker),
2. lists and filters trials by design, phase and whether results were posted,
3. reads a trial's posted results: group sizes, outcome values, confidence intervals, p-values,
4. runs Python in a sandbox when the question needs statistics,
5. writes a plain-language answer that weighs the evidence (large randomized trials over small or
   observational ones), cites every claim, and says when the database doesn't have the answer.

The page shows each research step live and keeps a "How I found this" panel under every answer.

## How it works

```
ClinicalTrials.gov API v2 ──► 10,269 COVID-19 study records ──┐
PubMed Central OA (AWS S3) ─► 1,630 full-text results papers ─┤
                                                              ▼
            chunk by section / outcome / table  (79,045 chunks)
                                                              ▼
       MedCPT Article Encoder embeddings + BM25 index  (LanceDB, one local file)
                                                              ▼
 question ─► Claude (tool use) ─► search_evidence ─► vector + keyword ─► fusion ─► MedCPT cross-encoder rerank
                              ├─► find_trials      (structured filters)
                              ├─► get_trial        (posted results, adverse events)
                              └─► code execution   (Anthropic sandbox, for statistics)
                                                              ▼
                     cited answer, streamed to a FastAPI + vanilla JS page
```

| Piece | Choice | Why |
|---|---|---|
| Embeddings | [MedCPT](https://huggingface.co/ncbi/MedCPT-Article-Encoder) (NCBI), run locally | Built for biomedical literature search; free |
| Vector store | LanceDB file | No database server, never pauses, no size limits |
| Search | Vector + BM25, reciprocal rank fusion, cross-encoder rerank | Drug names and trial IDs need exact matching; meaning needs vectors |
| Answering model | Claude Haiku 5.5 | Chosen by eval (below): ~$0.004 per question |
| Statistics | Claude code execution tool | Numbers are calculated, not recalled |

## Data notes

- **Registry → paper links are checked.** Registries sometimes list another trial's paper as a
  "result" (the Paxlovid EPIC-HR paper was listed under an unrelated cepharanthine trial). Papers
  are linked to the trial ID they state themselves; this corrected 107 of 1,630 links.
- **Retracted papers are excluded** using PubMed Central's retraction flag.
- **Coverage is limited to papers linked from the registry**, so some published trials are missing.
  The assistant is instructed to say so rather than fill gaps from memory.
- Paper licenses vary (CC BY, CC BY-NC, others). The built search database is not redistributed in
  this repo; rebuild it with the scripts below.

## Evaluation

A 20-question test set (`evals/cases.jsonl`) covers treatments, vaccines, statistics, finding
trials, honesty traps (a made-up trial, an out-of-scope question, a prompt-injection attempt) and
a personal medical question. Each answer is graded by code (are the cited IDs real? did it use code
for math?) and by a Claude judge against a reference (correct, honest about gaps, weighs evidence,
plain language).

| Model | Correct | Real citations | Cost / question | Typical time |
|---|---|---|---|---|
| Claude Opus 5.5 | 20/20* | 20/20 | ~$0.17 | 38 s |
| Claude Sonnet 5.5 | 17/20 | 20/20 | ~$0.055 | 19 s |
| Claude Haiku 5.5 | 19/20 | 20/20 | ~$0.004 | 23 s |

\*Opus also served as the judge, so its own scores may be lenient. Twenty questions run once is a
small sample: a second Haiku run scored 13/20, mostly from small numeric slips, so differences of a
few points are within noise. The eval did catch real bugs: a thread race in model loading that broke
search under concurrency, and the mislinked papers described above.

Run it: `python -m evals.run_eval --variant v1 --model claude-haiku-5-5`

## Running a public demo safely

- daily spend budget (default $0.50/day), then the chat pauses until tomorrow
- 10 questions per visitor and 20 per IP address per day
- optional Cloudflare Turnstile human check
- 500-character questions, 6 turns per conversation, 8 research steps per answer
- the API key stays on the server; the Anthropic Console workspace spend limit is the final backstop
- example questions on the page are pre-answered and cost nothing

## Run it locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt torch
echo "ANTHROPIC_API_KEY=..." > .env

# build the data (about an hour, mostly embedding on a laptop GPU)
.venv/bin/python scripts/fetch_trials.py       # registry records
.venv/bin/python scripts/profile_trials.py     # linked PubMed IDs
.venv/bin/python scripts/map_pmc.py            # PubMed -> PubMed Central IDs
.venv/bin/python scripts/fetch_pmc_oa.py       # open-access full text
.venv/bin/python scripts/build_chunks.py       # chunks + trial table
.venv/bin/python scripts/embed.py              # MedCPT embeddings
.venv/bin/python scripts/build_index.py        # LanceDB tables + keyword index

.venv/bin/uvicorn app.server:app --port 8770   # then open http://127.0.0.1:8770
```

`python -m app.chat "your question"` runs the assistant in the terminal.

## Layout

```
app/        agent (Claude loop + prompt), tools, retrieval, server, spend guard, web page
scripts/    data collection and index building
evals/      test cases and runner; results in .claude/hillclimb/covid-qa/
deploy/     Hugging Face Space container files
```

## License

Code: MIT (see `LICENSE`). Trial records are public domain (ClinicalTrials.gov); papers keep their
own licenses.
