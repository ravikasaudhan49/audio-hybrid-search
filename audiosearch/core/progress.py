"""Progress reporting for long-running services (Observer).

A service reports stages through a `Reporter` and stays unaware of who listens: the API's
background jobs (backend/jobs.py) record them for polling, the CLI prints them, and the
default does nothing.
"""
from typing import Protocol


class Reporter(Protocol):
    def stage(self, name: str, progress: float | None = None) -> None: ...

    def progress(self, value: float, note: str | None = None) -> None: ...

    def log(self, msg: str) -> None: ...


class NullReporter:
    def stage(self, name: str, progress: float | None = None) -> None:
        pass

    def progress(self, value: float, note: str | None = None) -> None:
        pass

    def log(self, msg: str) -> None:
        pass


class PrintReporter:
    """Stages and notes on stdout (the CLI)."""

    def stage(self, name: str, progress: float | None = None) -> None:
        print(f"-> {name}")

    def progress(self, value: float, note: str | None = None) -> None:
        if note:
            print(f"   {note}")

    def log(self, msg: str) -> None:
        print(f"   {msg}")
