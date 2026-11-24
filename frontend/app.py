"""Streamlit frontend.  Run:  streamlit run frontend/app.py   (backend must be running)

A pure HTTP client of the backend (CLAUDE.md rule 1): it imports nothing from the library
except shared logging. Layout:
  client.py      HTTP client for the API
  sidebar.py     collection picker, API status -> UiContext
  components.py  result cards, transcripts, job progress
  views/         one module per page, each with render(ctx)
"""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import client as client_module  # noqa: E402
from components import inject_css  # noqa: E402
from sidebar import render_sidebar  # noqa: E402
from views import about, evaluation, library, search, upload  # noqa: E402

from audiosearch.core import log as logs  # noqa: E402  (shared logging setup only; no DB or model access)

logs.setup("ui")

st.set_page_config(page_title="Audio Transcript Search", page_icon="🎧", layout="wide")
inject_css()


@st.cache_resource
def _client(version: float) -> client_module.Client:
    return client_module.Client()


def api() -> client_module.Client:
    """One API client per version of client.py: a new instance after the file changes, so a
    cached client never has stale methods during development."""
    return _client(Path(client_module.__file__).stat().st_mtime)


ctx = render_sidebar(api())


# st.navigation needs distinct named callables; each passes the context to its view.
def search_page():
    search.render(ctx)


def upload_page():
    upload.render(ctx)


def library_page():
    library.render(ctx)


def evaluation_page():
    evaluation.render(ctx)


def about_page():
    about.render(ctx)


pg = st.navigation([st.Page(search_page, title="Search", icon="🔎", default=True),
                    st.Page(upload_page, title="Upload & ingest", icon="⬆️", url_path="upload"),
                    st.Page(library_page, title="Library", icon="📚", url_path="library"),
                    st.Page(evaluation_page, title="Evaluation", icon="📊", url_path="evaluation"),
                    st.Page(about_page, title="How it works", icon="🧩", url_path="how-it-works")])
pg.run()
