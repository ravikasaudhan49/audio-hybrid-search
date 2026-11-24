-- Per-collection tables. Run with search_path = <collection schema>, public (see db.py);
-- the vector / pg_trgm extensions and the collections registry live in public.

CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,           -- e.g. "ep432"
    title       TEXT NOT NULL,
    audio_path  TEXT NOT NULL,
    duration_s  REAL,
    source      TEXT,
    -- 'golden' = evaluation dataset from the manifest; 'upload' = added through the UI.
    -- Evaluation only searches golden files, so demo uploads can't skew the numbers.
    origin      TEXT NOT NULL DEFAULT 'golden'
);

CREATE TABLE IF NOT EXISTS speakers (
    id       SERIAL PRIMARY KEY,
    file_id  TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    label    INT  NOT NULL,                 -- diarization label (0/1)
    name     TEXT NOT NULL,
    UNIQUE (file_id, label)
);

-- One row per CHILD chunk (~128 tokens, one speaker) = the retrieval unit.
-- The row also carries its PARENT (~512 tokens of whole turns: question + answer),
-- shared by all children of that parent; parent_id is the dedup key at query time.
CREATE TABLE IF NOT EXISTS segments (
    id              SERIAL PRIMARY KEY,
    file_id         TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    speaker_id      INT  NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    seq             INT  NOT NULL,          -- child order within file
    start_s         REAL NOT NULL,
    end_s           REAL NOT NULL,
    child_text      TEXT NOT NULL,
    child_tokens    INT  NOT NULL,
    words           JSONB NOT NULL,         -- [[word, start_s, end_s], ...] for exact hit timestamps
    parent_id       TEXT NOT NULL,          -- "<file_id>:p<NN>", unique across the corpus
    parent_start_s  REAL NOT NULL,
    parent_end_s    REAL NOT NULL,
    parent_text     TEXT NOT NULL,          -- "Host: ...\nGuest: ..." dialogue
    parent_tokens   INT  NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', child_text)) STORED,
    UNIQUE (file_id, seq)
);
CREATE INDEX IF NOT EXISTS segments_tsv_idx    ON segments USING GIN (tsv);
CREATE INDEX IF NOT EXISTS segments_trgm_idx   ON segments USING GIN (child_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS segments_parent_idx ON segments (parent_id);

-- Embeddings (of child_text) live in their own table so several models can be compared.
-- Dimension is left open; each model gets a partial HNSW expression index with its own dimension.
CREATE TABLE IF NOT EXISTS embeddings (
    segment_id  INT  NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedding   vector NOT NULL,
    PRIMARY KEY (segment_id, model)
);

-- ---------- episode timeline (added in place; existing rows and embeddings are kept) ----------
-- Clips are cut from longer episodes. All times above are relative to the stored CLIP (what
-- the local player plays); source_offset_s is where the clip starts in the original episode,
-- and the episode_* columns are derived from it, so the two timelines can never disagree.
-- Results display episode time. audio_url points at the original episode (dummy for now).
ALTER TABLE files    ADD COLUMN IF NOT EXISTS audio_url       TEXT;
ALTER TABLE files    ADD COLUMN IF NOT EXISTS source_offset_s REAL NOT NULL DEFAULT 0;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS audio_url       TEXT;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS source_offset_s REAL NOT NULL DEFAULT 0;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS episode_start_s REAL
    GENERATED ALWAYS AS (start_s + source_offset_s) STORED;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS episode_end_s REAL
    GENERATED ALWAYS AS (end_s + source_offset_s) STORED;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS parent_episode_start_s REAL
    GENERATED ALWAYS AS (parent_start_s + source_offset_s) STORED;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS parent_episode_end_s REAL
    GENERATED ALWAYS AS (parent_end_s + source_offset_s) STORED;

-- ---------- speaker role (host | guest), so "what did the guest say" can be answered ----------
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS role TEXT;
