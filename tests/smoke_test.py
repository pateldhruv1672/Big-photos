"""
Smoke Tests — Distributed Photo Intelligence Platform

Checks that all services are reachable and return sane responses.
Run with: python tests/smoke_test.py [--strict]

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

BACKEND_URL   = os.environ.get("BACKEND_URL", "http://localhost:8080")
RAY_SERVE_URL = os.environ.get("RAY_SERVE_URL", "http://ray-head:8000")
HDFS_UI_URL   = os.environ.get("HDFS_UI_URL", "http://namenode:9870")
RAY_DASH_URL  = os.environ.get("RAY_DASH_URL", "http://ray-head:8265")

TIMEOUT = 10  # seconds

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def ok(name: str, detail: str = "") -> bool:
    print(f"  {GREEN}✓{RESET} {name}" + (f" — {detail}" if detail else ""))
    return True


def fail(name: str, detail: str = "") -> bool:
    print(f"  {RED}✗{RESET} {name}" + (f" — {detail}" if detail else ""))
    return False


def warn(name: str, detail: str = "") -> None:
    print(f"  {YELLOW}~{RESET} {name} (optional)" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------ #
# Individual checks                                                     #
# ------------------------------------------------------------------ #

def check_backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
        body = r.json()
        assert r.status_code == 200, f"status {r.status_code}"
        assert body.get("status") == "ok", f"body: {body}"
        return ok("backend /health", f"kafka={body.get('kafka')}")
    except Exception as e:
        return fail("backend /health", str(e))


def check_backend_gallery() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/gallery?limit=5", timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        body = r.json()
        assert "photos" in body, f"no photos key: {body}"
        return ok("backend /gallery", f"{body.get('total', 0)} total photos")
    except Exception as e:
        return fail("backend /gallery", str(e))


def check_backend_search() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/search?q=nature&k=5", timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        body = r.json()
        assert "results" in body, f"no results key: {body}"
        return ok("backend /search", f"{len(body.get('results', []))} hits via {body.get('method', '?')}")
    except Exception as e:
        return fail("backend /search", str(e))


def check_backend_stories() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/stories", timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        body = r.json()
        assert "stories" in body, f"no stories key: {body}"
        return ok("backend /stories", f"{len(body.get('stories', []))} stories")
    except Exception as e:
        return fail("backend /stories", str(e))


def check_backend_metrics() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/metrics", timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        body = r.json()
        return ok("backend /metrics",
                  f"enriched={body.get('total_images_enriched', 0)}, "
                  f"hnsw={body.get('hnsw_index_size', 0)}")
    except Exception as e:
        return fail("backend /metrics", str(e))


def check_ray_serve() -> bool:
    try:
        r = requests.get(f"{RAY_SERVE_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        return ok("ray-serve /health")
    except Exception as e:
        return fail("ray-serve /health", str(e))


def check_ray_serve_predict() -> bool:
    try:
        payload = {
            "image_id": "smoke_test_001",
            "image_uri": "hdfs://namenode:9000/photos/raw/images/im0.jpg",
            "caption": "a photo of a mountain landscape",
            "labels": ["mountain", "nature", "landscape"],
        }
        r = requests.post(f"{RAY_SERVE_URL}/predict", json=payload, timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        body = r.json()
        assert "predicted_category" in body, f"missing predicted_category: {body}"
        assert "confidence" in body, f"missing confidence: {body}"
        return ok("ray-serve /predict",
                  f"category={body['predicted_category']}, confidence={body['confidence']:.2f}")
    except Exception as e:
        return fail("ray-serve /predict", str(e))


def check_hdfs_ui() -> bool:
    try:
        r = requests.get(HDFS_UI_URL, timeout=TIMEOUT)
        assert r.status_code == 200, f"status {r.status_code}"
        return ok("HDFS NameNode UI")
    except Exception as e:
        warn("HDFS NameNode UI", str(e))
        return True  # optional


def check_ray_dashboard() -> bool:
    try:
        r = requests.get(RAY_DASH_URL, timeout=TIMEOUT)
        assert r.status_code == 200
        return ok("Ray Dashboard")
    except Exception as e:
        warn("Ray Dashboard", str(e))
        return True  # optional


def check_local_eda_outputs() -> bool:
    # Spec (Part 1) required outputs + label_distribution added per architecture diagram
    required = [
        Path("/app/outputs/eda/tag_frequency.csv"),
        Path("/app/outputs/eda/image_dimensions.csv"),
        Path("/app/outputs/eda/dataset_summary.json"),
        Path("/app/outputs/eda/label_distribution.csv"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return fail("EDA outputs", f"missing: {missing}")
    summary = json.loads(Path("/app/outputs/eda/dataset_summary.json").read_text())
    # Optional chart files (warn, don't fail)
    charts = ["tag_frequency.png", "label_distribution.png", "image_size_histogram.png"]
    missing_charts = [c for c in charts if not Path(f"/app/outputs/eda/{c}").exists()]
    if missing_charts:
        warn("EDA charts", f"missing: {missing_charts} (install matplotlib to generate)")
    return ok("EDA outputs", f"num_images={summary.get('num_images', '?')}, tags={summary.get('unique_tags', '?')}")


def check_model_exists() -> bool:
    # Check for MobileNet V3 Small (.pt) first, then legacy sklearn (.pkl)
    model_path = Path("/app/models/image_classifier.pt")
    if not model_path.exists():
        model_path = Path("/app/models/image_classifier.pkl")
    if model_path.exists():
        size_kb = model_path.stat().st_size // 1024
        return ok("classifier model", f"{size_kb} KB")
    return fail("classifier model", "not found at /app/models/image_classifier.pkl")


def check_hnsw_index() -> bool:
    idx_path = Path("/app/outputs/hnsw_index.bin")
    map_path = Path("/app/outputs/id_map.json")
    if idx_path.exists() and map_path.exists():
        try:
            id_map = json.loads(map_path.read_text())
            return ok("HNSW index", f"{len(id_map)} vectors indexed")
        except Exception as e:
            return fail("HNSW index", str(e))
    return fail("HNSW index", "index or id_map not found")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Platform smoke tests")
    parser.add_argument("--strict", action="store_true",
                        help="Fail even on optional checks")
    args = parser.parse_args()

    print("\n=== Distributed Photo Intelligence — Smoke Tests ===\n")

    # Critical checks (must pass)
    critical = [
        check_backend_health,
        check_backend_gallery,
        check_backend_search,
        check_backend_stories,
        check_backend_metrics,
        check_ray_serve,
        check_ray_serve_predict,
    ]

    # File-system checks
    fs_checks = [
        check_local_eda_outputs,
        check_model_exists,
        check_hnsw_index,
    ]

    # Optional / infrastructure checks
    optional = [
        check_hdfs_ui,
        check_ray_dashboard,
    ]

    failed = []
    print("[ Service Endpoints ]")
    for fn in critical:
        if not fn():
            failed.append(fn.__name__)

    print("\n[ File-System Artifacts ]")
    for fn in fs_checks:
        if not fn():
            failed.append(fn.__name__)

    print("\n[ Infrastructure (optional) ]")
    for fn in optional:
        fn()  # warnings only, not counted as failures

    print()
    if failed:
        print(f"{RED}FAILED{RESET}: {len(failed)} check(s) → {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"{GREEN}All checks passed.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
