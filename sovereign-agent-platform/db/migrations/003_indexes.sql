CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_documents_scope ON documents(scope);
CREATE INDEX idx_events_trace_id ON events(trace_id);
CREATE INDEX idx_events_created_at ON events(created_at);
CREATE INDEX idx_policy_decisions_trace_id ON policy_decisions(trace_id);
CREATE INDEX idx_jobs_status ON jobs(status);
