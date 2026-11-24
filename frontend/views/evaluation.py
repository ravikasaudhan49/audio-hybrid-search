import streamlit as st
from client import ApiError
from components import follow_job
from context import UiContext

METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]


def render(ctx: UiContext) -> None:
    api = ctx.api
    st.header("Evaluation")
    st.caption("Recall@k and MRR on the labeled golden query set (golden files only). "
               "A hit = same file and overlapping time span (±2 s). "
               f"The golden labels describe the **nasa** collection; this runs on **{ctx.collection}**.")
    rerank = st.toggle("Include rerank and rerank + MMR configurations",
                       help=f"Uses {ctx.reranker}. Trial Cohere keys allow 10 rerank calls/min, so this waits "
                            "out the rate limit; scores are cached, so re-runs are fast.")
    if st.button(f"Run evaluation ({ctx.health['embedding_model']})"):
        follow_job(api, api.eval_run(rerank, ctx.collection), "Evaluating")
    try:
        report = api.eval_results()
    except ApiError:
        st.info("No results yet. Label queries in data/queries.json, then run the evaluation.")
        return
    pct = st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")
    st.subheader("Overall")
    st.dataframe(
        [{"configuration": label, **{m: r["overall"][m] for m in METRICS},
          "p50 ms": round(r["latency_ms"]["p50"]), "p95 ms": round(r["latency_ms"]["p95"]),
          "speaker acc.": r.get("speaker_accuracy")} for label, r in report.items()],
        hide_index=True, use_container_width=True, column_config={m: pct for m in [*METRICS, "speaker acc."]})
    st.subheader("Recall@5 by query type")
    types = sorted({t for r in report.values() for t in r["by_type"]})
    st.dataframe([{"configuration": label, **{t: r["by_type"].get(t, {}).get("recall@5") for t in types}}
                  for label, r in report.items()],
                 hide_index=True, use_container_width=True, column_config={t: pct for t in types})
    st.subheader("Misses (not found in top 10)")
    qs = {q["id"]: q for q in api.queries()}
    for label, r in report.items():
        if r["misses"]:
            with st.expander(f"{label}: {len(r['misses'])} missed"):
                for qid in r["misses"]:
                    q = qs.get(qid, {})
                    st.markdown(f"- `{qid}` ({q.get('type', '?')}) {q.get('query', '')}")
