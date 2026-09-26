# ADR-0003: Parent-child chunking (128 / 512 tokens)

**Status:** accepted · **Decided by:** project owner
**Context.** v1 (~30 s speaker-turn windows) separated questions from answers and produced 2-second
question chunks.

**Options.** A 512-token child / 2,000-token parent split was proposed. A 9-minute file is ~1,750
tokens, so the parent would be the whole file (one result per file) and children would mix
speakers.

**Decision.** Children ≤ 128 tokens (tiktoken cl100k, 25 overlap) inside one speaker's turn, which
is the retrieval unit with an exact speaker and timestamp; parents ≤ 512 tokens of whole turns,
used as context, for reranking and as the dedup key (`parent_id`). Backchannels get no child.

**Consequences.** ~25 children and 4–6 parents per clip. Grouping by parent can pick a sibling child
as a parent's representative; the ungrouped configuration is kept in the evaluation for comparison.
