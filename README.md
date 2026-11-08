# COVID Trials AI

A chat assistant that answers questions about COVID-19 using clinical trial
registry records and the open-access papers that report their results.
Every answer cites its sources.

Portfolio project. Research information only, not medical advice.

## Data sources

| Source | What | Count |
|---|---|---|
| [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) | Every study with condition "COVID-19" | 10,269 studies (1,096 with posted results) |
| ClinicalTrials.gov references | PubMed papers tagged as the trial's results | 2,883 papers |
| [PMC Open Access on AWS](https://registry.opendata.aws/ncbi-pmc/) | Full text of those papers that are open access | 1,630 papers covering 611 trials |

Each paper keeps its license (`license_code`) and retraction flag
(`is_retracted`) from the PMC metadata.

## Rebuild the data

```bash
python3 scripts/fetch_trials.py     # registry records -> data/raw/
python3 scripts/profile_trials.py   # summary + linked PMIDs
python3 scripts/map_pmc.py          # PMID -> PMCID -> data/trial_papers.json
python3 scripts/fetch_pmc_oa.py     # open-access full text -> data/pmc/
```
