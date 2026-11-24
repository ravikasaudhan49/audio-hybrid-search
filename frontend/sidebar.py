"""Sidebar: collection picker (+ create), API status, embedding model. Returns the page context."""
import streamlit as st
from client import ApiError, Client
from context import UiContext


def _collection_picker(api: Client) -> str:
    try:
        collections = api.collections()
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
                api.create_collection(new_name.strip(), new_desc or None)
                st.session_state.pop("collection", None)
                st.session_state["collection_to_select"] = new_name.strip()
                st.rerun()
            except ApiError as e:
                st.error(str(e))
    return collection


def _embedding_caption(h: dict) -> None:
    model = h["embedding_model"]
    if h["embeddings"].get(model):
        st.caption(f"**Embeddings:** {model} · {h['embedding_dims'][model]}-dim · "
                   f"{h['embeddings'][model]} of {h['segments']} chunks indexed")
    else:
        st.caption(f"**Embeddings:** {model} · nothing indexed yet in this collection")


def render_sidebar(api: Client) -> UiContext:
    with st.sidebar:
        st.title("🎧 Audio Search")
        collection = _collection_picker(api)
        try:
            h = api.health(collection)
        except ApiError as e:
            st.error(str(e))
            st.stop()
        st.caption(f"API connected · {h['files']} files · {h['segments']} segments")
        _embedding_caption(h)
        st.caption(f"**Reranker:** {h['reranker']}")
        st.caption(f"[API docs]({api.base_url}/docs)")
    return UiContext(api=api, health=h, collection=collection)
