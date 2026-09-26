---
name: golden-eval
description: Add or change golden (labeled) queries, rebuild the labels from transcript anchors, run the recall@k evaluation and report results honestly. Use when the golden dataset, retrieval pipeline or thresholds change.
---

# Golden-set evaluation workflow

Codified from the evaluation process the owner defined during development.

## 1. Write or change queries: `data/golden_spec.json`
Each answerable query needs:
- `id` (qNN), `type` (keyword | phrase | entity | semantic | speaker | role | typo), `query`, `file` (clip id)
- `anchor`: a phrase copied **verbatim** from that clip's transcript, inside the answering passage.

Rules:
- Semantic queries must paraphrase: avoid the transcript's own words.
- Negative queries (`type: negative`, no anchor) must be clearly off-topic for the whole collection.
- Cover every query type for every clip; include speaker and role queries ("what did X / the guest say…").

## 2. Build the labels (no API calls)
```bash
.venv\Scripts\python -m eval.build_golden
```
It must report 0 "not found" anchors. Labels = answer passage (same speaker, ±30 s around the
anchor, spanning backchannels) in episode time, plus the speaker name.

## 3. Estimate cost, then run
New queries → 1 Gemini embedding each; each changed candidate set → 1 Cohere rerank (10/min).
Tell the owner the expected number of calls, then:
```bash
.venv\Scripts\python -m audiosearch eval --rerank
.venv\Scripts\python -m pytest tests/test_recall.py -q
```

## 4. Investigate every miss before reporting
For each query in `misses` / `false_answers` (see `eval/results.json`), inspect what was returned
(`python -m audiosearch search "<query>" --context`, cached so no API call):
- **Retrieval bug** → fix the code, add a regression test.
- **Label too narrow** (a correct passage scored as a miss) → fix the labelling *method* in
  `eval/build_golden.py`, never a single label; re-run everything.
- **Genuine limitation** → report it; do **not** tune thresholds to hide it.

## 5. Report
Update the results tables in `README.md` §5 and §7 and add an `AGENT_LOG.md` entry with the
before/after numbers and what caused any change.
