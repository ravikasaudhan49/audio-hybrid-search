# ADR-0008: Golden labels anchored to transcript quotes

**Status:** accepted · **Decided by:** project owner
**Context.** Hand-typed timestamps are unverifiable and drift with chunking.

**Decision.** Each query stores a verbatim transcript anchor; `eval/build_golden.py` derives the
answer passage (same speaker, ±30 s, spanning backchannels) in episode time plus the speaker.
Phrase queries accept any occurrence. Labels are corrected only by changing the *method*, never
per query.

**Consequences.** Two method corrections, each triggered by a correct result scored as a miss,
logged with before/after numbers. Queries were written by the builders after reading the
transcripts, so recall@5 saturates; blind labeling is future work.
