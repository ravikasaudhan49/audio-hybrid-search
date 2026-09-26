"""Streamlit frontend.  Run:  streamlit run frontend/app.py   (backend must be running)"""
import hashlib
import html
import importlib
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
import client as client_module

from audiosearch import log as logs  # shared logging setup only; no DB or model access

# Streamlit re-runs app.py on every change but keeps imported modules; reload the API client so
# an edited client.py is never paired with a newer app.py (stale-method errors).
importlib.reload(client_module)
ApiError, Client = client_module.ApiError, client_module.Client

logs.setup("ui")

st.set_page_config(page_title="Audio Transcript Search", page_icon="🎧", layout="wide")

# Two speakers per file; identity is carried by a colored bar, never by text color.
SPEAKER_COLORS = ["#2a6fdb", "#e0782f"]
METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]

st.markdown("""<style>
.turn{border-left:4px solid var(--c);padding:4px 10px;margin:6px 0}
.turn .who{font-size:12px;opacity:.75}
.card{border:1px solid rgba(128,128,128,.3);border-radius:8px;padding:10px 12px;margin:8px 0}
.card .meta{font-size:13px;opacity:.8;margin-bottom:4px}
.card mark{background:#ffe066;color:#111;padding:0 2px;border-radius:2px}
.why{font-size:12px;opacity:.65;margin-top:4px}
.ok{color:#1a7f37;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def _client(version: float) -> Client:
    return Client()


def api() -> Client:
    """One API client per version of client.py (a new instance after the module changes)."""
    return _client(Path(client_module.__file__).stat().st_mtime)


def fmt_ts(s: float) -> str:
    h, rem = divmod(int(s or 0), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def highlight(s: str) -> str:
    return re.sub(r"\[\[(.+?)\]\]", r"<mark>\1</mark>", html.escape(s))


def snippet(highlighted: str, width: int) -> str:
    words = highlighted.split()
    first = next((i for i, w in enumerate(words) if w.startswith("[[")), 0)
    lo = max(0, first - width // 3)
    return ("… " if lo else "") + " ".join(words[lo:lo + width]) + (" …" if lo + width < len(words) else "")


def render_turns(turns: list[dict], names: dict | None = None):
    parts = []
    for t in turns:
        name = (names or {}).get(t["speaker"]) or t.get("speaker_name") or f"Speaker {t['speaker']}"
        parts.append(f'<div class="turn" style="--c:{SPEAKER_COLORS[t["speaker"] % 2]}">'
                     f'<div class="who">{html.escape(name)} · {fmt_ts(t["start"])}–{fmt_ts(t["end"])}</div>'
                     f'{html.escape(t["text"])}</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def follow_job(job: dict, title: str) -> dict:
    """Show a live status box for a backend job until it finishes."""
    with st.status(title, expanded=True) as box:
        bar = st.progress(0.0)
        seen = 0

        def update(j):
            nonlocal seen
            for line in j["log"][seen:]:
                st.write(line)
            seen = len(j["log"])
            bar.progress(min(1.0, j["progress"]), text=j["stage"])

        job = api().wait(job["id"], update)
        if job["status"] == "error":
            box.update(label=f"{title}: failed", state="error")
            st.error(job["error"])
        else:
            box.update(label=f"{title}: done", state="complete")
    return job


# ---------- sidebar ----------

with st.sidebar:
    st.title("🎧 Audio Search")
    try:
        collections = api().collections()
    except ApiError as e:
        st.error(str(e))
        st.stop()
    # A collection is an isolated dataset (its own tables); upload and search both work in it.
    names = [c["name"] for c in collections]
    counts = {c["name"]: c for c in collections}
    if "collection_to_select" in st.session_state:       # just created: select it
        st.session_state["collection"] = st.session_state.pop("collection_to_select")
    if st.session_state.get("collection") not in names:
        st.session_state["collection"] = "nasa" if "nasa" in names else names[0]
    collection = st.selectbox(
        "Collection", names, key="collection",
        format_func=lambda n: f"{n}  ({counts[n]['files']} files · {counts[n]['segments']} chunks)",
        help="Each collection has its own tables with the same columns. Uploads go into, and searches read "
             "from, the selected collection only.")
    with st.expander("➕ New collection"):
        new_name = st.text_input("Name", placeholder="e.g. acme_calls", key="new_collection_name",
                                 help="1-40 lowercase letters, digits or _, starting with a letter")
        new_desc = st.text_input("Description (optional)", key="new_collection_desc")
        if st.button("Create collection", disabled=not new_name):
            try:
                api().create_collection(new_name.strip(), new_desc or None)
                st.session_state.pop("collection", None)
                st.session_state["collection_to_select"] = new_name.strip()
                st.rerun()
            except ApiError as e:
                st.error(str(e))
    try:
        h = api().health(collection)
    except ApiError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"API connected · {h['files']} files · {h['segments']} segments")
    model_of = {"gemini": "gemini-embedding-001", "bge-base": "bge-base-en-v1.5", "bge-small": "bge-small-en-v1.5"}
    # Only models whose vectors are actually in the index can serve vector search.
    indexed = [e for e in h["embedders"] if h["embeddings"].get(model_of.get(e, e))] or [h["default_embedder"]]
    if len(indexed) > 1:
        embedder = st.selectbox("Embedding model", indexed,
                                index=indexed.index(h["default_embedder"]) if h["default_embedder"] in indexed else 0)
    else:
        embedder = indexed[0]
    model = model_of.get(embedder, embedder)
    if h["embeddings"].get(model):
        st.caption(f"**Embeddings:** {model} · {h['embedding_dims'][model]}-dim · "
                   f"{h['embeddings'][model]} of {h['segments']} chunks indexed")
    else:
        st.caption(f"**Embeddings:** {model} · nothing indexed yet in this collection")
    st.caption(f"**Reranker:** {h['reranker']}")
    st.caption(f"[API docs]({api().base_url}/docs)")


# ---------- pages ----------

def page_search():
    st.header("Search")
    st.caption('Search exact words, a "quoted phrase", or a concept. Results show file, speaker and timestamp.')
    st.caption(f"Collection: **{collection}**")
    files = {f["id"]: f["title"] for f in api().files(collection)}

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
    rerank = c6.toggle(f"Rerank ({h['reranker']})", value=h["rerank_default"],
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
    mmr_lambda = 0.6
    if mmr:
        mmr_lambda = st.slider("MMR balance (λ)", 0.0, 1.0, 0.6, 0.05,
                               help="1.0 = pure relevance (same as off) · lower = more diverse results")
    opts = dict(speaker=speaker, file_id=file_id, rerank=rerank, mmr=mmr, mmr_lambda=mmr_lambda, group=group,
                embedder=embedder, role=role, auto_speaker=auto_speaker, threshold=threshold, collection=collection)

    if not q:
        queries = api().queries()
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
                    res = api().search(q, method, k, **(opts if full else {**opts, "rerank": False, "mmr": False}))
                    st.subheader(title + pipeline_label(res))
                    status_line(res)
                    show_results(res, compact=True)
        else:
            res = api().search(q, mode.lower(), k, answer=want_answer, **opts)
            status_line(res)
            show_answer(res)
            if res["results"]:
                st.markdown("#### Retrieved chunks")
            show_results(res)
    except ApiError as e:
        st.error(str(e))


def show_answer(res: dict):
    a = res.get("answer")
    if not a:
        return
    # Make [n] citations point at the numbered result cards below.
    text = re.sub(r"\[(\d+)\]", r"**[\1]**", a["text"])
    with st.container(border=True):
        st.markdown("#### Answer")
        st.markdown(text)
        st.caption(f"Generated by {a['model']} from the top results only"
                   + (f" · cites result(s) {', '.join(map(str, a['citations']))}" if a["citations"] else "")
                   + (" · cached" if a["cached"] else ""))


def pipeline_label(res: dict) -> str:
    return (" + rerank" if res["reranked"] else "") + (" + MMR" if res["mmr"] else "")


def status_line(res: dict):
    who = res["speaker_detected"] or (f"the {res['role_detected']}" if res["role_detected"] else None)
    if who:
        st.info(f"Detected **{who}** in the query → searching *“{res['search_text']}”* in their words only. "
                "(Turn off *Detect speaker in query* to search everyone.)")
    parts = [f"{len(res['results'])} results", f"{res['took_ms']:.0f} ms"]
    if res["below_threshold"]:
        parts.append(f"{res['below_threshold']} low-relevance hidden")
    if res["reranked"]:
        parts.append(f"reranked by {res['reranker']}")
    if res["mmr"]:
        parts.append("diversified with MMR")
    if res["golden_query"]:
        parts.append("golden query: ✅ marks labeled-relevant results")
    st.caption(" · ".join(parts))
    for note in res["notes"]:
        st.warning(note)


def show_results(res: dict, compact: bool = False):
    if not res["results"]:
        st.info("No results.")
    for i, r in enumerate(res["results"], 1):
        mark = ' <span class="ok">✅ relevant</span>' if r["relevant"] else ""
        why = " · ".join(f"{m} #{rank}" for m, rank in r["sources"].items())
        st.markdown(
            f'<div class="card"><div class="meta"><b>{i}. {html.escape(r["file_title"])}</b> ({r["file_id"]}) · '
            f'<b>{html.escape(r["speaker"])}</b> · ▶ <b>{r["timestamp"]}</b> '
            f'<span style="opacity:.6">({fmt_ts(r["episode_start_s"])}–{fmt_ts(r["episode_end_s"])} in episode)'
            f'</span>{mark}</div>'
            f'{highlight(snippet(r["highlighted"], 35 if compact else 70))}'
            f'<div class="why">via {why} · score {r["score"]:.4f} · parent {r["parent_id"]}'
            + (f' ({r["grouped"]} children matched)' if r["grouped"] > 1 else "") + '</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"Context: parent dialogue {fmt_ts(r['parent_episode_start_s'])}–"
                         f"{fmt_ts(r['parent_episode_end_s'])}"):
            st.text(r["parent_text"])
        with st.expander(f"Play from {r['timestamp']}"):
            # The stored clip is played, so seek with clip time (hit_s); the label shows episode time.
            st.audio(api().audio_src(r["stream_url"]), start_time=max(0, int(r["hit_s"]) - 1))
            if r["audio_url"]:
                st.caption(f"Source episode: {r['audio_url']}")


def page_upload():
    st.header("Upload & ingest")
    st.caption("Each pipeline stage runs on the backend as a job: normalize → transcribe + diarize → "
               "name speakers → chunk → embed + index.")
    st.info(f"Uploading into collection **{collection}** (change it in the sidebar).")
    up = st.file_uploader("Audio file with two speakers", type=["mp3", "wav", "m4a", "ogg", "flac"])
    title = st.text_input("Title", value=Path(up.name).stem if up else "")
    keyterms = st.text_input("Key terms (optional, comma separated)",
                             help="Names and jargon; the transcriber uses these to spell rare words correctly.")
    if not up:
        return
    # Same id the backend derives from the file content: the stored transcription result only
    # belongs to THIS file if the ids match (otherwise a previous upload would still be shown
    # and "Index" would store the previous file).
    fid = "up_" + hashlib.sha1(up.getvalue()).hexdigest()[:8]
    st.caption(f"Selected: **{up.name}** · {up.size / 1e6:.1f} MB · file id `{fid}`")

    if st.button("1 · Transcribe", type="primary"):
        try:
            job = api().upload(up.name, up.getvalue(), title, keyterms, collection)
        except ApiError as e:
            st.error(str(e))
            return
        job = follow_job(job, f"Transcribing {up.name}")
        if job["status"] == "done":
            st.session_state.upload = job["result"]

    result = st.session_state.get("upload")
    if not result:
        return
    if result["file_id"] != fid:
        st.info(f"The transcript shown before was for another file. Press **1 · Transcribe** to process "
                f"**{up.name}**.")
        return

    st.subheader("2 · Name the speakers")
    st.caption(f"{result.get('provider', 'transcriber')} returned speaker labels {result['raw_speaker_labels']}, "
               "cleaned to the two main "
               f"speakers · {len(result['turns'])} turns · audio {fmt_ts(result['duration_s'])}")
    names = {}
    host = st.radio("Who is the host?", (0, 1), horizontal=True, key=f"host_{fid}",
                    format_func=lambda s: f"Speaker {s}",
                    help="Stored as the speaker role, so queries like “what did the guest say …” work.")
    for spk, col in zip((0, 1), st.columns(2), strict=True):
        with col:
            identified = (result.get("speaker_names") or {}).get(str(spk))
            names[spk] = st.text_input(f"Speaker {spk}" + (" (identified by transcriber)" if identified else ""),
                                       value=identified or f"Speaker {spk + 1}", key=f"name_{fid}_{spk}")
            sample = result["samples"].get(str(spk)) or result["samples"].get(spk) or ""
            st.caption(f"“{sample}…”")

    st.subheader("3 · Diarized transcript")
    with st.expander("Full transcript", expanded=True):
        render_turns(result["turns"], names)

    st.subheader("4 · Chunks")
    st.caption(f"{len(result['parents'])} parents (~512 tokens of whole turns) → {len(result['children'])} "
               "children (~128 tokens, 25-token overlap, always one speaker). Children are embedded and searched; "
               "the parent is the context and dedup key.")
    st.dataframe([{"#": i, "parent": f"p{c['parent']:02d}", "speaker": names.get(c["speaker"]),
                   "start": fmt_ts(c["start"]), "end": fmt_ts(c["end"]), "tokens": c["tokens"], "text": c["text"]}
                  for i, c in enumerate(result["children"])],
                 hide_index=True, use_container_width=True, height=260)

    if st.button(f"5 · Index with {embedder}", type="primary"):
        try:
            job = api().index(fid, result["title"], names, embedder, host, collection)
        except ApiError as e:
            st.error(str(e))
            return
        if follow_job(job, "Indexing")["status"] == "done":
            st.success(f"Indexed “{result['title']}”. Try it on the Search page.")


def page_library():
    st.header("Library")
    st.caption(f"Collection: **{collection}**")
    files = api().files(collection)
    if not files:
        st.info("Nothing indexed yet. Run `python -m audiosearch ingest` or use Upload.")
        return
    st.dataframe([{"id": f["id"], "title": f["title"], "dataset": f["origin"], "length": fmt_ts(f["duration_s"]),
                   "speakers": " & ".join(f["speakers"]), "segments": f["segments"]} for f in files],
                 hide_index=True, use_container_width=True)
    pick = st.selectbox("View transcript", [f["id"] for f in files],
                        format_func=lambda i: next(f"{f['id']} · {f['title']}" for f in files if f["id"] == i))
    detail = api().file(pick, collection)
    st.audio(api().audio_src(detail["stream_url"]))
    if detail["audio_url"]:
        st.caption(f"Clip starts at {fmt_ts(detail['source_offset_s'])} of the source episode · {detail['audio_url']}")
    render_turns(detail["turns"])
    if detail["origin"] == "upload" and st.button(f"Delete {pick}"):
        api().delete_file(pick, collection)
        st.rerun()


def page_eval():
    st.header("Evaluation")
    st.caption("Recall@k and MRR on the labeled golden query set (golden files only). "
               "A hit = same file and overlapping time span (±2 s). "
               f"The golden labels describe the **nasa** collection; this runs on **{collection}**.")
    rerank = st.toggle("Include rerank and rerank + MMR configurations",
                       help=f"Uses {h['reranker']}. Trial Cohere keys allow 10 rerank calls/min, so this waits "
                            "out the rate limit; scores are cached, so re-runs are fast.")
    if st.button(f"Run evaluation ({embedder})"):
        follow_job(api().eval_run([embedder], rerank, collection), "Evaluating")
    try:
        report = api().eval_results()
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
    qs = {q["id"]: q for q in api().queries()}
    for label, r in report.items():
        if r["misses"]:
            with st.expander(f"{label}: {len(r['misses'])} missed"):
                for qid in r["misses"]:
                    q = qs.get(qid, {})
                    st.markdown(f"- `{qid}` ({q.get('type', '?')}) {q.get('query', '')}")


def page_about():
    st.header("How it works")
    st.markdown(f"""
```
Streamlit UI ──HTTP──► FastAPI backend ({api().base_url}/docs)
                          │  owns the Postgres pool + loaded embedding models; slow work runs as jobs
                          ▼
audio ─► ffmpeg (16 kHz mono) ─► AssemblyAI (words + timestamps + diarization + speaker NAMES)
      ─► speaker cleanup (exactly 2, flip smoothing) ─► turns
      ─► parents: ~512 tokens of whole turns ─► children: ~128 tokens in one speaker's turn (25 overlap)
      ─► Postgres: segments (text, speaker, start/end, word timings)
             ├─ tsvector + GIN        → keyword / phrase search (term coverage, then ts_rank_cd)
             ├─ pg_trgm GIN           → typo-tolerant search
             └─ pgvector HNSW (cosine) per embedding model → semantic search
query ─► keyword ∥ fuzzy ∥ vector  (pipelined, one DB round trip)
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


pg = st.navigation([st.Page(page_search, title="Search", icon="🔎", default=True),
                    st.Page(page_upload, title="Upload & ingest", icon="⬆️", url_path="upload"),
                    st.Page(page_library, title="Library", icon="📚", url_path="library"),
                    st.Page(page_eval, title="Evaluation", icon="📊", url_path="evaluation"),
                    st.Page(page_about, title="How it works", icon="🧩", url_path="how-it-works")])
pg.run()
