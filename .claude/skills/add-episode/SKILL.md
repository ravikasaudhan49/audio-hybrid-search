---
name: add-episode
description: Add a new podcast episode to a collection, either as a golden 8-10 minute clip (for evaluation) or as a normal full-length upload. Covers window selection, speaker names and roles, and ingestion with minimal API use.
---

# Adding an episode

Codified from the ingestion process the owner defined during development.

## A. Normal upload (searchable, not part of evaluation)
UI → **Upload & ingest** (choose the collection first), or:
```bash
.venv\Scripts\python -m audiosearch add path\to\episode.mp3 --collection nasa --title "..." --host 0
```
Check the speaker names the transcriber identified (fix artefacts such as "Name - 1") and which
label is the host.

## B. Golden clip (evaluation)
1. **Transcribe the full episode once** (upload it as in A, or use an existing transcript). Never
   re-transcribe a clip.
2. **Pick a 9-minute window** from the transcript: two-speaker back-and-forth (most turns), host
   share ≥ ~15% where possible, skip the intro (first ~3 min) and outro credits.
3. **Add a manifest entry** in `data/manifest.json`:
   ```json
   {"id": "epNNN", "title": "...", "audio": "epNNN.mp3", "host": "<host name>",
    "audio_url": "<source episode url>", "source_url": "...", "license": "...",
    "clip": {"source": "audio/<full episode>.mp3", "start": "00:12:30", "duration": 540,
             "transcript_from": "<id of the full-episode transcript>",
             "source_speakers": {"1": "<corrected name if needed>"}}}
   ```
   Every file in a collection must be a **unique pair** of speakers.
4. **Ingest** (cuts the clip audio and slices its transcript; only embedding calls are made):
   ```bash
   .venv\Scripts\python -m audiosearch ingest
   ```
5. Verify names and roles: `python -m audiosearch preview epNNN`, then add queries with the
   `golden-eval` skill.
