# Start the Proxy

Choose one method below to start the proxy in a separate terminal.

---

## 1. Standard venv

If you created a virtual environment with `python3 -m venv .venv`:

```bash
.venv/bin/python start_proxy.py
```

## 2. uv

If you are using `uv` for dependency management:

```bash
uv run python start_proxy.py
```

## 3. Docker Compose

If you prefer running via Docker:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

---

## Verify it's running

Once started, you can check the health endpoint:

```bash
curl http://localhost:8083/health
```

Or open the dashboard: `http://localhost:8083/dashboard`

> If you changed the `PORT` in `.env`, replace `8083` with your configured port.

## Port already in use?

Check what's using it:

```bash
# macOS / Linux
lsof -i :8083
```

Or find and kill the process:

```bash
kill $(lsof -t -i :8083)
```
