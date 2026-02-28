INSERT INTO events (trace_id, event_type, payload, payload_sha256)
VALUES (
  'seed-000',
  'system_init',
  '{"message": "platform initialized"}',
  encode(digest('{"message": "platform initialized"}', 'sha256'), 'hex')
);
