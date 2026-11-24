import streamlit as st
from client import ApiError
from components import pipeline_label, show_answer, show_results, status_line
from context import UiContext


def render(ctx: UiContext) -> None:
    api, h = ctx.api, ctx.health
    st.header("Search")
    st.caption('Search exact words, a "quoted phrase", or a concept. Results show file, speaker and timestamp.')
    st.caption(f"Collection: **{ctx.collection}**")
    files = {f["id"]: f["title"] for f in api.files(ctx.collection)}

    # The query row is a form: the search runs on the Search button (or Enter), not on every keystroke.
    with st.form("search_form", border=False):
        c1, c2, c3, c4, c0 = st.columns([5, 2, 2, 1, 1])
        typed = c1.text_input("Query", value=st.session_state.get("query", ""), label_visibility="collapsed",
                              placeholder="e.g. what did Kranz say about accountability")
        mode = c2.selectbox("Mode", ["Hybrid", "Compare methods", "Keyword", "Vector", "Fuzzy"],
                            label_visibility="collapsed")
        file_id = c3.selectbox("File", [None, *files], label_visibility="collapsed",
                               format_func=lambda f: "All files" if f is None else f"{f} · {files[f]}")
        k = c4.number_input("k", 1, 20, 5, label_visibility="collapsed")
        if c0.form_submit_button("🔎 Search", type="primary", use_container_width=True):
            st.session_state["query"] = typed.strip()
    q = st.session_state.get("query", "")
    c5, c5b, c6, c7, c8 = st.columns([3, 2, 2, 2, 2])
    speaker = c5.text_input("Speaker filter (optional)", placeholder="speaker name contains…") or None
    role = c5b.selectbox("Role", [None, "host", "guest"], format_func=lambda r: "Any role" if r is None else r,
                         help="Only the host's or the guest's words.")
    rerank = c6.toggle(f"Rerank ({ctx.reranker})", value=h["rerank_default"],
                       help="A reranker model reads the query and each of the top 30 candidates together "
                            "and re-orders them by relevance.")
    mmr = c7.toggle("MMR diversify", value=False,
                    help="Not satisfied with the results? Maximal Marginal Relevance re-picks the top results "
                         "to be relevant AND different from each other: near-duplicate chunks drop out and "
                         "other parts of the conversation surface.")
    group = c8.toggle("Group by parent", value=True,
                      help="Children of the same ~512-token parent collapse into one result (the best-matching "
                           "child); the reranker then judges the whole parent dialogue.")
    c9, c10, c11, _ = st.columns([2, 2, 2, 3])
    auto_speaker = c9.toggle("Detect speaker in query", value=True,
                             help='"what did Jordan say about X" searches only Gary Jordan\'s words for X; '
                                  '"what did the guest say …" only the guest\'s.')
    threshold = c10.toggle("Hide low-relevance results", value=True,
                           help="Drops results whose reranker score is below the relevance threshold, "
                                "even inside the top k. Needs Rerank on.")
    want_answer = c11.toggle("Summary answer", value=True,
                             help="Gemini Flash-Lite answers from the top results only and cites them as "
                                  "[1], [2] … (one API call per new query; repeats are cached).")
    mmr_lambda = None   # server default
    if mmr:
        mmr_lambda = st.slider("MMR balance (λ)", 0.0, 1.0, 0.6, 0.05,
                               help="1.0 = pure relevance (same as off) · lower = more diverse results")
    opts = dict(speaker=speaker, file_id=file_id, rerank=rerank, mmr=mmr, mmr_lambda=mmr_lambda, group=group,
                role=role, auto_speaker=auto_speaker, threshold=threshold,
                collection=ctx.collection)

    if not q:
        queries = api.queries()
        if queries:
            with st.expander("Example labeled queries from the golden set (results get ✅ when relevant)"):
                for item in queries[:25]:
                    st.markdown(f"- `{item.get('type', '')}` {item['query']}")
        return
    try:
        if mode == "Compare methods":
            # Single methods are shown raw; the hybrid column applies the rerank/MMR options.
            columns = [("Keyword", "keyword", False), ("Vector", "vector", False), ("Hybrid", "hybrid", True)]
            for col, (title, method, full) in zip(st.columns(3), columns, strict=True):
                with col:
                    res = api.search(q, method, k, **(opts if full else {**opts, "rerank": False, "mmr": False}))
                    st.subheader(title + pipeline_label(res))
                    status_line(res)
                    show_results(api, res, compact=True)
        else:
            res = api.search(q, mode.lower(), k, answer=want_answer, **opts)
            status_line(res)
            show_answer(res)
            if res["results"]:
                st.markdown("#### Retrieved chunks")
            show_results(api, res)
    except ApiError as e:
        st.error(str(e))
