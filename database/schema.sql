PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    meeting_date TEXT NOT NULL,              -- ISO date
    timezone TEXT NOT NULL DEFAULT 'UTC',
    source_type TEXT NOT NULL,               -- audio | transcript | seed
    status TEXT NOT NULL DEFAULT 'created',  -- created | transcribed | analysed | failed
    duration_s REAL,
    media_path TEXT,
    is_seed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_s REAL,
    end_s REAL,
    speaker TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_seg_meeting ON transcript_segments(meeting_id, idx);

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    display_name TEXT
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    config_json TEXT,
    stats_json TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES analysis_runs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    method TEXT NOT NULL,                    -- llm | extractive
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES analysis_runs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    evidence_idx TEXT,                       -- JSON list of segment idx
    evidence_quote TEXT,
    confidence REAL,
    source TEXT NOT NULL,                    -- classical | llm
    fallback_reason TEXT,
    status TEXT NOT NULL DEFAULT 'suggested' -- suggested | needs_review | confirmed | rejected
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES analysis_runs(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    owner TEXT,
    deadline_text TEXT,
    deadline_date TEXT,
    evidence_idx TEXT,
    evidence_quote TEXT,
    confidence REAL,
    source TEXT NOT NULL,
    fallback_reason TEXT,
    status TEXT NOT NULL DEFAULT 'suggested'
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    seg_start INTEGER,
    seg_end INTEGER,
    start_s REAL,
    end_s REAL,
    speakers TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_chunk_meeting ON chunks(meeting_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    scope_json TEXT
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT,
    supported INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS mom_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    content_md TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calendar_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    action_item_id INTEGER REFERENCES action_items(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    event_date TEXT NOT NULL,
    description TEXT,
    uid TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'proposed',  -- proposed | confirmed
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS email_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    recipients TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,                     -- draft_prepared (nothing is ever sent by the app)
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    event TEXT NOT NULL,
    meeting_id INTEGER,
    detail TEXT
);
