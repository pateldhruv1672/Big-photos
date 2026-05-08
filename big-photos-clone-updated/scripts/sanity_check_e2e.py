#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import requests
from requests import RequestException


def _get_json(url: str, timeout: int = 20, retries: int = 6, sleep_s: float = 2.0) -> Dict[str, Any]:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (RequestException, ValueError) as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _get_status(url: str, timeout: int = 20) -> int:
    r = requests.get(url, timeout=timeout)
    return r.status_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8001")
    parser.add_argument("--frontend-url", default="http://localhost:5173")
    parser.add_argument("--gallery-limit", type=int, default=20)
    parser.add_argument("--thumb-check-count", type=int, default=8)
    args = parser.parse_args()

    api = args.api_base.rstrip("/")
    report: Dict[str, Any] = {"api_base": api, "frontend_url": args.frontend_url}

    health = _get_json(f"{api}/health", timeout=20, retries=8, sleep_s=2.0)
    report["health"] = health
    if health.get("status") != "ok":
        raise SystemExit("Backend health status is not ok")

    gallery = _get_json(f"{api}/api/gallery?limit={args.gallery_limit}", timeout=30, retries=5, sleep_s=1.5)
    items: List[Dict[str, Any]] = gallery.get("items") or []
    report["gallery_count"] = len(items)
    report["gallery_total_known"] = int(gallery.get("total_known", 0) or 0)
    if not items:
        raise SystemExit("Gallery returned zero items")

    stories = _get_json(f"{api}/api/stories", timeout=20, retries=3, sleep_s=1.0)
    report["stories_count"] = len(stories.get("stories") or [])

    thumb_statuses: List[int] = []
    for item in items[: max(1, int(args.thumb_check_count))]:
        thumb_url = item.get("thumbnail_url")
        if not thumb_url:
            continue
        status = _get_status(f"{api}{thumb_url}", timeout=20)
        thumb_statuses.append(status)
    report["thumbnail_statuses"] = thumb_statuses
    ok_thumbs = sum(1 for x in thumb_statuses if x == 200)
    report["thumbnail_ok"] = ok_thumbs
    report["thumbnail_checked"] = len(thumb_statuses)
    if len(thumb_statuses) == 0 or ok_thumbs == 0:
        raise SystemExit("Thumbnail sanity failed: no successful thumbnail responses")

    frontend_status = _get_status(args.frontend_url, timeout=20)
    report["frontend_status"] = frontend_status
    if frontend_status != 200:
        raise SystemExit(f"Frontend status expected 200, got {frontend_status}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
