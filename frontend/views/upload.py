import hashlib
from pathlib import Path

import streamlit as st
from client import ApiError
from components import fmt_ts, follow_job, render_turns
from context import UiContext


def render(ctx: UiContext) -> None:
    api = ctx.api
    st.header("Upload & ingest")
    st.caption("Each pipeline stage runs on the backend as a job: normalize → transcribe + diarize → "
               "name speakers → chunk → embed + index.")
    st.info(f"Uploading into collection **{ctx.collection}** (change it in the sidebar).")
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
            job = api.upload(up.name, up.getvalue(), title, keyterms, ctx.collection)
        except ApiError as e:
            st.error(str(e))
            return
        job = follow_job(api, job, f"Transcribing {up.name}")
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

    if st.button(f"5 · Index with {ctx.health['embedding_model']}", type="primary"):
        try:
            job = api.index(fid, result["title"], names, host, ctx.collection)
        except ApiError as e:
            st.error(str(e))
            return
        if follow_job(api, job, "Indexing")["status"] == "done":
            st.success(f"Indexed “{result['title']}”. Try it on the Search page.")
