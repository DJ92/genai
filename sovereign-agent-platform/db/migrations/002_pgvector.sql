CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE embeddings DROP COLUMN embedding;
ALTER TABLE embeddings ADD COLUMN embedding vector(768);
