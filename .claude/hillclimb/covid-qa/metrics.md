# COVID Q&A eval metrics

- **correct** (judge): conclusion matches the reference; no factual errors about the trials.
- **citations_valid** (code): every NCT/PMC ID in the answer appeared in a tool result during that turn.
- **honest_gaps** (judge): candid about missing data; no unretrieved claims presented as database findings.
- **weighs_evidence** (judge): separates strong (large randomized, blinded) from weak evidence.
- **plain_language** (judge): bottom line first, followable by a non-expert.
- **used_code** (code, statistics cases only; blank elsewhere): at least one code-execution call.

Judge: claude-opus-5-5, structured output, one call per answer. Opus also answers in the
baseline variant, so its baseline grades may be lenient.
