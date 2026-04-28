import asyncio
import os
import time

import httpx

if __name__ != "__main__":
    import pytest

    if os.environ.get("RUN_PROXY_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "Integration tests are disabled by default. Set RUN_PROXY_INTEGRATION_TESTS=1 to enable.",
            allow_module_level=True,
        )
    pytestmark = pytest.mark.asyncio


async def test_log_rotation():
    # Test the endpoint with large payloads to trigger log rotation
    url = "http://localhost:8083/api/event_logging/batch"

    # Create a large payload (about 20KB)
    large_data = "x" * 20000

    async with httpx.AsyncClient() as client:
        for i in range(1000):  # Send 1000 requests
            payload = [
                {
                    "event_type": "log_rotation_test",
                    "data": {"id": i, "payload": large_data, "timestamp": time.time()},
                }
            ]

            response = await client.post(url, json=payload)
            print(f"Request {i+1}: {response.status_code} - {response.json()}")

            if i % 100 == 0:
                print(f"Sent {i+1} requests")

            # Small delay to avoid overwhelming
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(test_log_rotation())
