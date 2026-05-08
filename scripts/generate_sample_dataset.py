"""
Generate Sample Dataset

Produces synthetic JPEG images paired with REAL tags from the MIRFLICKR dataset.

Tag source priority:
  1. data/tags.zip  — the real MIRFLICKR tags archive (tags/{id%10}/{id}.txt)
  2. data/mirflickr/tags/  — pre-extracted directory with same layout
  3. Fallback: small curated tag vocabulary (used only if neither source is present)

Images are always synthetic (coloured squares) since we only have the tags file,
not the full 12 GB image archive.  The pipeline treats these as stand-ins for the
real MIRFLICKR images; swap in real JPEGs at data/mirflickr/images/ any time and
re-run `make part1` — nothing else needs to change.

Usage:
  python scripts/generate_sample_dataset.py           # default 100 images
  python scripts/generate_sample_dataset.py --n 500   # custom sample size
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100,
                    help="Number of sample images to generate (default: 100)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
N = args.n

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data/mirflickr")
IMG_DIR   = DATA_DIR / "images"
TAGS_JSON = DATA_DIR / "tags.json"
TAGS_ZIP  = Path("data/tags.zip")
TAGS_DIR  = DATA_DIR / "tags"   # pre-extracted layout: tags/{id%10}/{id}.txt

IMG_DIR.mkdir(parents=True, exist_ok=True)

_GENERATED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _clear_generated_images(img_dir: Path) -> None:
    """Remove files from a prior run so Spark sees exactly N new images (folder kept)."""
    if not img_dir.is_dir():
        return
    for path in img_dir.iterdir():
        if path.is_file() and path.suffix.lower() in _GENERATED_IMAGE_SUFFIXES:
            path.unlink()


_clear_generated_images(IMG_DIR)

# ─── Location vocabulary (inferred from real tags) ────────────────────────────

LOCATION_KEYWORDS = {
    "San Francisco": {"sanfrancisco", "sf", "goldengatebridge", "bayarea", "california"},
    "New York":      {"newyork", "nyc", "manhattan", "brooklyn", "timessquare"},
    "London":        {"london", "uk", "england", "bigben", "britishmuseum"},
    "Tokyo":         {"tokyo", "japan", "shinjuku", "shibuya", "akihabara"},
    "Paris":         {"paris", "france", "eiffeltower", "louvre", "montmartre"},
    "Beach":         {"beach", "ocean", "sea", "surf", "waves", "sand", "coast"},
    "Mountain":      {"mountain", "hiking", "alps", "summit", "peak", "snow"},
    "Forest":        {"forest", "trees", "woods", "nature", "green"},
}

FALLBACK_TAG_POOLS = {
    "nature":  ["mountain", "forest", "lake", "sky", "sunset", "nature", "landscape", "clouds"],
    "city":    ["city", "urban", "street", "building", "skyline", "bridge", "architecture"],
    "people":  ["people", "portrait", "face", "crowd", "person", "smile"],
    "travel":  ["travel", "vacation", "tourism", "holiday", "trip", "adventure"],
    "food":    ["food", "restaurant", "coffee", "meal", "cuisine", "bakery"],
    "event":   ["concert", "festival", "party", "sport", "wedding", "event"],
    "beach":   ["beach", "ocean", "surf", "sand", "sea", "waves", "coast"],
}

LOCATIONS = list(LOCATION_KEYWORDS.keys())


def _ascii_safe_draw_label(text: str, max_len: int = 72) -> str:
    """Strip non-ASCII for PIL's default font (bitmap/latin-1); avoids UnicodeEncodeError."""
    if not text:
        return ""
    safe = text.encode("ascii", errors="ignore").decode("ascii")
    safe = " ".join(safe.split())
    if len(safe) > max_len:
        safe = safe[: max_len - 3].rstrip() + "..."
    return safe


def _infer_location(tags):
    tag_set = {t.lower().replace(" ", "") for t in tags}
    for loc, keywords in LOCATION_KEYWORDS.items():
        if tag_set & keywords:
            return loc
    return random.choice(LOCATIONS)


def _fallback_tags():
    pool = random.choice(list(FALLBACK_TAG_POOLS.values()))
    return random.sample(pool, min(4, len(pool)))


# ─── Read tags from zip ───────────────────────────────────────────────────────

def _bucket(img_id):
    """MIRFLICKR bucket: tags/{img_id // 10000}/{img_id}.txt"""
    return img_id // 10000


def _read_tags_from_zip(zip_path, image_ids):
    result = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        name_set = set(zf.namelist())
        for img_id in image_ids:
            member = f"tags/{_bucket(img_id)}/{img_id}.txt"
            if member in name_set:
                try:
                    text = zf.read(member).decode("utf-8", errors="ignore")
                    tags = [t.strip() for t in text.splitlines() if t.strip()]
                    result[img_id] = tags
                except Exception:
                    result[img_id] = []
            else:
                result[img_id] = []
    return result


def _read_tags_from_dir(tags_dir, image_ids):
    result = {}
    for img_id in image_ids:
        tag_file = tags_dir / str(_bucket(img_id)) / f"{img_id}.txt"
        if tag_file.exists():
            tags = [t.strip() for t in tag_file.read_text(errors="ignore").splitlines() if t.strip()]
            result[img_id] = tags
        else:
            result[img_id] = []
    return result


# ─── Determine tag source and sample IDs ─────────────────────────────────────

sample_ids = random.sample(range(1_000_000), N)

print(f"Generating {N} sample images with real MIRFLICKR tags...")

if TAGS_ZIP.exists():
    print(f"  Source: {TAGS_ZIP}")
    id_to_tags = _read_tags_from_zip(TAGS_ZIP, sample_ids)
    source = "MIRFLICKR_REAL"
elif TAGS_DIR.exists():
    print(f"  Source: {TAGS_DIR} (pre-extracted)")
    id_to_tags = _read_tags_from_dir(TAGS_DIR, sample_ids)
    source = "MIRFLICKR_REAL"
else:
    print("  No tags source found — using fallback vocabulary.")
    id_to_tags = {i: _fallback_tags() for i in sample_ids}
    source = "MIRFLICKR_SAMPLE"

# ─── Generate images + build metadata ────────────────────────────────────────

PALETTE = [
    (70, 130, 180), (34, 139, 34),  (205, 92, 92),  (255, 165, 0),
    (147, 112, 219),(32, 178, 170), (210, 105, 30),  (119, 136, 153),
]

metadata = {}

for idx, img_id in enumerate(sample_ids):
    tags = id_to_tags.get(img_id) or _fallback_tags()
    location = _infer_location(tags)

    # Synthetic image: coloured square with ID + first tag
    color = PALETTE[idx % len(PALETTE)]
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 118), f"im{img_id}", fill=(255, 255, 255))
    draw.text((10, 138), _ascii_safe_draw_label(tags[0] if tags else ""), fill=(220, 220, 220))

    file_name = f"im{img_id}.jpg"
    img.save(IMG_DIR / file_name)

    # Spread taken_at across years/months to make temporal EDA interesting
    year  = 2009 + (idx % 10)
    month = (idx % 12) + 1
    day   = (idx % 28) + 1

    metadata[f"im{img_id}"] = {
        "mirflickr_id": img_id,
        "tags":         tags,
        "location":     location,
        "taken_at":     f"{year}-{month:02d}-{day:02d}T{(idx % 12) + 8:02d}:00:00",
        "source":       source,
    }

    if (idx + 1) % 20 == 0 or (idx + 1) == N:
        print(f"  {idx + 1}/{N}")

TAGS_JSON.write_text(json.dumps(metadata, indent=2))

print(f"\nDone.")
print(f"  Images   → {IMG_DIR}  ({N} files)")
print(f"  Metadata → {TAGS_JSON}")
print(f"  Source   : {source}")
print(f"  Sample   : im{sample_ids[0]} → {id_to_tags.get(sample_ids[0], [])[:5]}")
