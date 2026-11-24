import streamlit as st
from components import fmt_ts, render_turns
from context import UiContext


def render(ctx: UiContext) -> None:
    api = ctx.api
    st.header("Library")
    st.caption(f"Collection: **{ctx.collection}**")
    files = api.files(ctx.collection)
    if not files:
        st.info("Nothing indexed yet. Run `python -m audiosearch ingest` or use Upload.")
        return
    st.dataframe([{"id": f["id"], "title": f["title"], "dataset": f["origin"], "length": fmt_ts(f["duration_s"]),
                   "speakers": " & ".join(f["speakers"]), "segments": f["segments"]} for f in files],
                 hide_index=True, use_container_width=True)
    pick = st.selectbox("View transcript", [f["id"] for f in files],
                        format_func=lambda i: next(f"{f['id']} · {f['title']}" for f in files if f["id"] == i))
    detail = api.file(pick, ctx.collection)
    st.audio(api.audio_src(detail["stream_url"]))
    if detail["audio_url"]:
        st.caption(f"Clip starts at {fmt_ts(detail['source_offset_s'])} of the source episode · {detail['audio_url']}")
    render_turns(detail["turns"])
    if detail["origin"] == "upload" and st.button(f"Delete {pick}"):
        api.delete_file(pick, ctx.collection)
        st.rerun()
