#!/usr/bin/env python3
"""Manual auth diagnostic for OpenAI-compatible providers.

Usage:
  ./venv/bin/python tests/auth_test.py
  ./venv/bin/python tests/auth_test.py --model zai-org/GLM-4.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import httpx
from dotenv import load_dotenv


def _redact(value: str, keep: int = 4) -> str:
    if not value:
        return "<empty>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{'*' * max(len(value) - keep, 1)}{value[-keep:]}"


def _get_custom_headers() -> Dict[str, str]:
    custom_headers: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("CUSTOM_HEADER_"):
            name = key[len("CUSTOM_HEADER_") :].replace("_", "-")
            if name:
                custom_headers[name] = value
    return custom_headers


def _choose_model(explicit_model: str | None) -> str | None:
    if explicit_model:
        return explicit_model
    for key in ("SMALL_MODEL", "MIDDLE_MODEL", "BIG_MODEL"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _print_response(response: httpx.Response) -> None:
    body = response.text.strip()
    if len(body) > 400:
        body = f"{body[:400]}...<truncated>"
    print(f"status={response.status_code} body={body}")


def _diagnose_401() -> None:
    print("\n401 diagnosis hints:")
    print("1) OPENAI_API_KEY is invalid/expired or from the wrong project/account.")
    print("2) OPENAI_API_KEY includes an unwanted 'Bearer ' prefix.")
    print("3) CUSTOM_HEADER_AUTHORIZATION or CUSTOM_HEADER_X_API_KEY overrides auth.")
    print("4) OPENAI_BASE_URL does not match the key's expected endpoint.")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Manual provider auth diagnostic")
    parser.add_argument("--model", default=None, help="Model to test")
    parser.add_argument(
        "--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    model = _choose_model(args.model)

    if not api_key:
        print("ERROR: OPENAI_API_KEY is empty")
        return 1
    if not base_url:
        print("ERROR: OPENAI_BASE_URL is empty")
        return 1
    if not model:
        print("ERROR: no model found. Set SMALL_MODEL/MIDDLE_MODEL/BIG_MODEL or use --model")
        return 1

    custom_headers = _get_custom_headers()
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(custom_headers)

    print("Auth diagnostic config:")
    print(f"- base_url={base_url}")
    print(f"- model={model}")
    print(f"- api_key={_redact(api_key)}")
    print(f"- custom_header_keys={sorted(custom_headers.keys())}")

    try:
        with httpx.Client(timeout=args.timeout) as client:
            print("\n[1/2] GET /models")
            models_resp = client.get(f"{base_url}/models", headers=headers)
            _print_response(models_resp)
            if models_resp.status_code == 401:
                _diagnose_401()
                return 1

            print("\n[2/2] POST /chat/completions")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with: OK"}],
                "max_tokens": 8,
                "temperature": 0,
            }
            chat_resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            _print_response(chat_resp)

            if chat_resp.status_code == 401:
                _diagnose_401()
                return 1
            if chat_resp.status_code >= 400:
                print("\nAuth appears to pass, but request still failed.")
                print(
                    "This can indicate model name/permissions/payload issues rather than key auth."
                )
                return 1

            data = chat_resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "<no-content>")
            print("\nSUCCESS: provider auth and completion call both succeeded.")
            print(f"model_reply={json.dumps(content)}")
            return 0

    except httpx.RequestError as exc:
        print(f"Network/request error: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - guardrail for manual tool
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
