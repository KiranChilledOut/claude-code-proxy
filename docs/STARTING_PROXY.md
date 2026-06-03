# Starting the Proxy

Three ways to start the proxy. Pick the one that matches how you set up the project.

---

## 1. Standard venv

If you used `python3 -m venv .venv` (or the TUI created it for you):

```bash
.venv/bin/python start_proxy.py
```

## 2. uv

If you created the environment with `uv venv` (or the TUI chose uv):

```bash
uv run python start_proxy.py
```

## 3. Docker Compose

```bash
docker compose up --build -d
```

To rebuild after code changes:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

---

## Verify it's running

Open the dashboard: http://localhost:8083/dashboard

Or check the health endpoint:

```bash
curl http://localhost:8083/health
```

## Port already in use?

Check what's listening:

```bash
# macOS / Linux
lsof -i :8083

# Or find and kill the process
kill $(lsof -t -i :8083)
```

If you changed `PORT` in `.env`, use that number instead of `8083`.
