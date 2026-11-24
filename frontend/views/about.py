import streamlit as st
from context import UiContext


def render(ctx: UiContext) -> None:
    st.header("How it works")
    st.markdown(f"""
```
Streamlit UI ──HTTP──► FastAPI backend ({ctx.api.base_url}/docs)
                          │  owns the Postgres pool + loaded embedding models; slow work runs as jobs
                          ▼
audio ─► ffmpeg (16 kHz mono) ─► AssemblyAI (words + timestamps + diarization + speaker NAMES)
      ─► speaker cleanup (exactly 2, flip smoothing) ─► turns
      ─► parents: ~512 tokens of whole turns ─► children: ~128 tokens in one speaker's turn (25 overlap)
      ─► Postgres: segments (text, speaker, start/end, word timings)
             ├─ tsvector + GIN        → keyword / phrase search (term coverage, then ts_rank_cd)
             ├─ pg_trgm GIN           → typo-tolerant search
             └─ pgvector HNSW (cosine) per embedding model → semantic search
query ─► keyword, fuzzy, vector  (back to back on one pooled connection, a few ms)
      ─► weighted Reciprocal Rank Fusion ─► top 30 candidates
      ─► rerank: Cohere Rerank reads (query, chunk) pairs together        [on by default]
      ─► MMR: relevant AND unlike results already picked                [toggle]
      ─► file · speaker · exact word timestamp · highlighted snippet
```
- **Why chunk by speaker turn?** Every result has exactly one speaker, so attribution is unambiguous.
- **Why hybrid?** Keyword search nails exact names and phrases; vectors catch paraphrases. RRF merges
  them by rank only, so no score calibration between the two is needed.
- **Why word timings?** A keyword hit jumps to the second the word was spoken, not the chunk start.
- **Why rerank?** Fusion only sees ranks. A reranker reads the query and the chunk together, so it can tell
  "talks about X" from "mentions X in passing". It is too slow for the whole corpus, so it only re-orders the top 30.
- **Why MMR?** Neighbouring chunks overlap and a conversation repeats itself, so the top results can say the same
  thing several times. MMR trades a little relevance for coverage, using the stored chunk embeddings (no API call).
""")
