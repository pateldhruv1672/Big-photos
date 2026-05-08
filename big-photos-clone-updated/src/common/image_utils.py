import hashlib
import io
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

CATEGORIES = ["travel", "nature", "people", "food", "city", "indoor", "animal", "sports"]
LOCATIONS = ["San Francisco", "New York", "Beach", "Mountain", "City", "Park", "Lake"]
LABEL_BANK = {
    "travel": ["travel", "landmark", "outdoor"],
    "nature": ["nature", "trees", "sky"],
    "people": ["people", "portrait", "group"],
    "food": ["food", "restaurant", "meal"],
    "city": ["city", "street", "building"],
    "indoor": ["indoor", "room", "object"],
    "animal": ["animal", "pet", "wildlife"],
    "sports": ["sports", "action", "field"],
}


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def image_id_from_path(path: str) -> str:
    name = os.path.basename(path).split(".")[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    suffix = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{suffix}"


def infer_fallback_category(seed_text: str) -> str:
    return CATEGORIES[stable_hash(seed_text) % len(CATEGORIES)]


def infer_fallback_labels(seed_text: str) -> List[str]:
    category = infer_fallback_category(seed_text)
    labels = list(LABEL_BANK[category])
    extra = CATEGORIES[(stable_hash(seed_text + "extra") + 2) % len(CATEGORIES)]
    if extra not in labels:
        labels.append(extra)
    return labels


def infer_fallback_location(seed_text: str) -> str:
    return LOCATIONS[stable_hash(seed_text + "loc") % len(LOCATIONS)]


def infer_fallback_taken_at(seed_text: str) -> datetime:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    days = stable_hash(seed_text + "date") % 120
    minutes = stable_hash(seed_text + "minute") % (24 * 60)
    return base + timedelta(days=days, minutes=minutes)


def image_info_and_thumbnail(data: bytes, max_size: int = 256) -> Tuple[Dict[str, int | str | bool], bytes]:
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        fmt = img.format or "JPEG"
        thumb = img.convert("RGB")
        thumb.thumbnail((max_size, max_size))
        out = io.BytesIO()
        thumb.save(out, format="JPEG", quality=82, optimize=True)
        return {"width": int(width), "height": int(height), "format": fmt, "is_valid_image": True}, out.getvalue()


def make_caption(category: str, labels: List[str], file_name: str) -> str:
    label_text = ", ".join(labels[:3])
    return f"A {category} photo with visual cues such as {label_text}. Source file: {file_name}."
