CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE documents (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_uri    TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'personal',
    title         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    offset_start    INTEGER,
    offset_end      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE embeddings (
    chunk_id            UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding_model_id  TEXT NOT NULL,
    embedding           BYTEA,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, embedding_model_id)
);

CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued','running','completed','failed','cancelled')),
    priority        INTEGER NOT NULL DEFAULT 0,
    workflow_name   TEXT NOT NULL,
    state           JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id          UUID REFERENCES jobs(id),
    trace_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    payload_sha256  TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE RULE events_no_update AS ON UPDATE TO events DO INSTEAD NOTHING;
CREATE RULE events_no_delete AS ON DELETE TO events DO INSTEAD NOTHING;

CREATE TABLE policy_decisions (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id               TEXT NOT NULL,
    subject                TEXT NOT NULL,
    action                 TEXT NOT NULL,
    resource               TEXT NOT NULL,
    decision               TEXT NOT NULL CHECK (decision IN ('allow','deny','needs_approval')),
    reason                 TEXT,
    policy_bundle_version  TEXT NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
