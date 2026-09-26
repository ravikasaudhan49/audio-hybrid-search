"""Thin HTTP client for the backend API. The frontend never touches the DB or models directly.

Every call is logged with its duration and the API's X-Request-ID, so a UI log line can be
matched to the backend's log lines for the same request.
"""
import logging
import os
import time

import httpx

log = logging.getLogger("frontend.client")
QUIET_PATHS = ("/jobs/", "/health")  # polled constantly: DEBUG only

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    pass


class Client:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=120)

    def _req(self, method: str, path: str, **kw):
        t0 = time.perf_counter()
        try:
            r = self.http.request(method, path, **kw)
        except httpx.ConnectError as e:
            log.error("%s %s -> backend unreachable at %s", method, path, self.base_url)
            raise ApiError(f"Backend not reachable at {self.base_url}. Start it with "
                           f"`python -m audiosearch serve`.") from e
        ms = (time.perf_counter() - t0) * 1000
        rid = r.headers.get("x-request-id", "-")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            log.warning("%s %s -> %d in %.0f ms (api request %s): %s", method, path, r.status_code, ms, rid, detail)
            raise ApiError(f"{r.status_code}: {detail}")
        level = logging.DEBUG if path.startswith(QUIET_PATHS) else logging.INFO
        log.log(level, "%s %s %s -> %d in %.0f ms (api request %s)", method, path,
                kw.get("params", ""), r.status_code, ms, rid)
        return r.json()

    def audio_src(self, relative_url: str) -> str:
        return self.base_url + relative_url

    def health(self, collection=None):
        return self._req("GET", "/health", params=_c(collection))

    def collections(self):
        return self._req("GET", "/collections")

    def create_collection(self, name: str, description: str | None = None):
        return self._req("POST", "/collections", json={"name": name, "description": description})

    def search(self, q, method="hybrid", k=5, speaker=None, file_id=None, rerank=True, mmr=False,
               mmr_lambda=0.7, group=True, embedder=None, role=None, auto_speaker=True, threshold=True,
               collection=None, answer=False):
        params = {"q": q, "method": method, "k": k, "rerank": rerank, "mmr": mmr, "mmr_lambda": mmr_lambda,
                  "group": group, "speaker": speaker, "file_id": file_id, "embedder": embedder, "role": role,
                  "auto_speaker": auto_speaker, "threshold": threshold, "collection": collection,
                  "answer": answer}
        return self._req("GET", "/search", params={k_: v for k_, v in params.items() if v is not None})

    def queries(self):
        return self._req("GET", "/queries")

    def files(self, collection=None):
        return self._req("GET", "/files", params=_c(collection))

    def file(self, file_id, collection=None):
        return self._req("GET", f"/files/{file_id}", params=_c(collection))

    def delete_file(self, file_id, collection=None):
        return self._req("DELETE", f"/files/{file_id}", params=_c(collection))

    def upload(self, name: str, data: bytes, title: str, keyterms: str, collection=None):
        return self._req("POST", "/uploads", files={"file": (name, data)},
                         data={"title": title, "keyterms": keyterms, "collection": collection or ""})

    def index(self, file_id, title, speakers: dict, embedder=None, host=0, collection=None):
        return self._req("POST", f"/uploads/{file_id}/index",
                         json={"title": title, "speakers": speakers, "embedder": embedder, "host": host,
                               "collection": collection})

    def job(self, job_id):
        return self._req("GET", f"/jobs/{job_id}")

    def wait(self, job_id, on_update=None, interval=0.7):
        """Poll a job until it finishes; `on_update(job)` is called on every poll."""
        while True:
            job = self.job(job_id)
            if on_update:
                on_update(job)
            if job["status"] != "running":
                return job
            time.sleep(interval)

    def eval_results(self):
        return self._req("GET", "/eval/results")

    def eval_run(self, embedders: list[str], rerank=False, collection=None):
        return self._req("POST", "/eval/run", params={"embedders": embedders, "rerank": rerank, **_c(collection)})


def _c(collection) -> dict:
    return {"collection": collection} if collection else {}
