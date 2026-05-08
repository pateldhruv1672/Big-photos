from __future__ import annotations

import json
import os
import sys

import requests

URL = os.getenv("RAY_SERVE_URL", "http://ray-head:8000/predict")


def main() -> int:
    try:
        resp = requests.post(URL, json={"image_id": "sample_001", "image_uri": "hdfs://namenode:9000/photos/raw/images/sample_001.jpg"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "predicted_category" in data
        assert "confidence" in data
        assert "labels" in data
        print(json.dumps(data, indent=2))
        return 0
    except Exception as exc:
        print(f"Ray Serve smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
