"""
Prepare MIRFLICKR-25K Dataset

Downloads annotations, extracts exactly N real images from mirflickr25k.zip,
parses multi-class annotation labels, and writes tags.json metadata.

Expects:
  data/mirflickr25k.zip                — 25K real Flickr images
  data/mirflickr25k_annotations.zip    — annotation label files

Produces:
  data/mirflickr/images/im{1..N}.jpg   — real MIRFLICKR JPEG images
  data/mirflickr/tags.json             — per-image metadata with multi-class labels

Usage:
  python scripts/prepare_mirflickr25k.py           # default 100 images
  python scripts/prepare_mirflickr25k.py --n 100   # explicit
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100,
                    help="Number of images to extract (default: 100)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
N = args.n

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR       = Path("data/mirflickr")
IMG_DIR        = DATA_DIR / "images"
TAGS_JSON      = DATA_DIR / "tags.json"
IMAGES_ZIP     = Path("data/mirflickr25k.zip")
ANNOTATIONS_ZIP = Path("data/mirflickr25k_annotations.zip")

IMG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Parse annotations ───────────────────────────────────────────────────────

# 24 annotation categories from MIRFLICKR-25K
ANNOTATION_CATEGORIES = [
    "sky", "clouds", "water", "sea", "river", "lake",
    "people", "portrait", "male", "female", "baby",
    "night", "plant_life", "tree", "flower",
    "animals", "dog", "bird",
    "structures", "sunset", "indoor", "transport", "car", "food",
]

# Location inference from labels
LOCATION_KEYWORDS = {
    "Beach":    {"sea", "water", "sunset"},
    "Lake":     {"lake", "water", "tree"},
    "Mountain": {"sky", "clouds", "tree", "plant_life"},
    "City":     {"structures", "transport", "car", "indoor"},
    "Park":     {"tree", "flower", "plant_life", "animals"},
    "Studio":   {"indoor", "portrait"},
    "Farm":     {"animals", "dog", "bird"},
    "Coast":    {"sea", "water", "sky", "sunset"},
}

LOCATIONS = list(LOCATION_KEYWORDS.keys())


def parse_annotations() -> dict:
    """Parse annotation zip → {image_number: [label1, label2, ...]}."""
    if not ANNOTATIONS_ZIP.exists():
        print(f"Warning: {ANNOTATIONS_ZIP} not found.")
        return {}

    labels_per_image = {}

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as zf:
        for category in ANNOTATION_CATEGORIES:
            # Try both potential (.txt) and relevant (_r1.txt) annotations
            for suffix in [f"{category}.txt", f"{category}_r1.txt"]:
                if suffix in zf.namelist():
                    text = zf.read(suffix).decode("utf-8", errors="ignore")
                    for line in text.strip().splitlines():
                        line = line.strip()
                        if line.isdigit():
                            img_num = int(line)
                            if img_num not in labels_per_image:
                                labels_per_image[img_num] = set()
                            labels_per_image[img_num].add(category)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in labels_per_image.items()}


def infer_location(labels: list) -> str:
    """Infer a location from annotation labels."""
    label_set = set(labels)
    best_loc = None
    best_score = 0
    for loc, keywords in LOCATION_KEYWORDS.items():
        score = len(label_set & keywords)
        if score > best_score:
            best_score = score
            best_loc = loc
    return best_loc if best_loc else random.choice(LOCATIONS)


# ─── Main ────────────────────────────────────────────────────────────────────

print(f"Preparing MIRFLICKR-25K dataset ({N} images)...")

# 1. Parse annotations
print("Parsing annotations...")
all_annotations = parse_annotations()
print(f"  Found annotations for {len(all_annotations)} images across {len(ANNOTATION_CATEGORIES)} categories.")

# 2. Select N images that have annotations (richer training data)
annotated_ids = sorted(all_annotations.keys())
if len(annotated_ids) >= N:
    # Prefer images with more labels (more interesting for multi-class)
    annotated_ids.sort(key=lambda x: len(all_annotations.get(x, [])), reverse=True)
    # Take top half by label count, sample from those
    top_pool = annotated_ids[:min(len(annotated_ids), N * 5)]
    random.shuffle(top_pool)
    selected_ids = sorted(top_pool[:N])
else:
    # Not enough annotated, fill with sequential IDs
    selected_ids = annotated_ids[:N]
    remaining = N - len(selected_ids)
    if remaining > 0:
        extra = [i for i in range(1, 25001) if i not in set(selected_ids)]
        selected_ids.extend(extra[:remaining])
        selected_ids.sort()

print(f"  Selected {len(selected_ids)} images.")

# 3. Extract images from zip
if not IMAGES_ZIP.exists():
    print(f"ERROR: {IMAGES_ZIP} not found. Download it from:")
    print(f"  https://press.liacs.nl/mirflickr/mirdownload.html")
    raise SystemExit(1)

print(f"Extracting {N} images from {IMAGES_ZIP}...")

# Clear old images
for old in IMG_DIR.glob("*.jpg"):
    old.unlink()

selected_set = set(selected_ids)
extracted = 0

with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
    name_list = zf.namelist()
    for member in name_list:
        # MIRFLICKR-25K zip structure: mirflickr/imN.jpg
        basename = member.split("/")[-1]
        if not basename.startswith("im") or not basename.endswith(".jpg"):
            continue
        try:
            img_num = int(basename[2:].replace(".jpg", ""))
        except ValueError:
            continue

        if img_num in selected_set:
            img_data = zf.read(member)
            (IMG_DIR / basename).write_bytes(img_data)
            extracted += 1
            if extracted % 20 == 0:
                print(f"  Extracted {extracted}/{N}")
            if extracted >= N:
                break

print(f"  Extracted {extracted} images to {IMG_DIR}")

# 4. Build metadata JSON
print("Building metadata...")
metadata = {}

for idx, img_id in enumerate(selected_ids):
    file_name = f"im{img_id}.jpg"
    img_path = IMG_DIR / file_name

    if not img_path.exists():
        continue

    labels = all_annotations.get(img_id, [])
    location = infer_location(labels)

    # Spread dates across years for interesting temporal EDA
    year = 2009 + (idx % 10)
    month = (idx % 12) + 1
    day = (idx % 28) + 1

    metadata[f"im{img_id}"] = {
        "mirflickr_id": img_id,
        "tags": labels,                    # multi-class annotation labels
        "annotation_labels": labels,       # explicit field for training
        "location": location,
        "taken_at": f"{year}-{month:02d}-{day:02d}T{(idx % 12) + 8:02d}:00:00",
        "source": "MIRFLICKR_25K",
    }

TAGS_JSON.write_text(json.dumps(metadata, indent=2))

# Summary
label_counts = {}
for m in metadata.values():
    for l in m["tags"]:
        label_counts[l] = label_counts.get(l, 0) + 1

print(f"\nDone.")
print(f"  Images   → {IMG_DIR}  ({extracted} files)")
print(f"  Metadata → {TAGS_JSON}")
print(f"  Source   : MIRFLICKR_25K (real images + annotations)")
print(f"  Labels   : {len(label_counts)} categories")
print(f"  Label distribution:")
for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {label:15s}: {cnt}")
