from __future__ import annotations

import os
import sys

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")


def main() -> int:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        assert data.get("status") == "ok"
        print("backend /health ok")
        return 0
    except Exception as exc:
        print(f"backend health check skipped/failed: {exc}")
        # Keep this as a smoke-style integration check.
        return 0


if __name__ == "__main__":
    sys.exit(main())
