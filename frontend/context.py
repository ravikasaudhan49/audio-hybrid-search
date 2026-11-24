"""What every page needs, chosen once per run in the sidebar and passed to the page explicitly
(instead of module-level globals)."""
from dataclasses import dataclass

from client import Client


@dataclass
class UiContext:
    api: Client
    health: dict          # GET /health for the selected collection
    collection: str

    @property
    def reranker(self) -> str:
        return self.health["reranker"]
