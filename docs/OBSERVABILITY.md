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
MODEL_PRICES_JSON='{"moonshotai/Kimi-K2.6":{"input_per_1m":0.95,"output_per_1m":4.00,"advertised_tok_s":60,"currency":"USD"}}'
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
