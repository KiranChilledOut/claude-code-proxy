# Observability Dashboard

The proxy includes an embedded dashboard at:

```text
http://localhost:8083/dashboard
```

It records request metadata, model routing, token usage, estimated cost, latency,
failures, and tool-call names into SQLite. Prompt and completion bodies are not
stored.

## Configuration

```bash
OBSERVABILITY_ENABLED=true
OBSERVABILITY_DB_PATH=/app/data/observability.sqlite3
OBSERVABILITY_QUEUE_SIZE=1000
OBSERVABILITY_STORE_TOOL_ARGS=false
MODEL_PRICES_JSON='{"zai-org/GLM-4.7-FP8":{"input_per_1m":0.30,"output_per_1m":1.20,"advertised_tok_s":36.8,"currency":"USD"}}'
```

Pricing is env-configured only. If a backend model is missing from
`MODEL_PRICES_JSON`, the dashboard still shows usage, latency, failures, and
tool calls, but cost is shown as not configured.

## Docker Persistence

`docker-compose.yml` bind-mounts `./data` to `/app/data`, so the SQLite database
lives under the repository root and survives container stop/start cycles:

```yaml
volumes:
  - ./data:/app/data
```

## Runtime Behavior

Observability writes use a bounded in-memory queue and a background SQLite
writer. If the queue fills, observability records are dropped instead of slowing
the proxy request path.
