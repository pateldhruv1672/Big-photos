#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(os.getenv("SAMPLE_DATA_DIR", "/app/data/sample_images"))
N = int(os.getenv("SAMPLE_IMAGE_COUNT", "120"))
CATEGORIES = ["travel", "nature", "people", "food", "city", "indoor", "animal", "sports"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N):
        cat = CATEGORIES[i % len(CATEGORIES)]
        img = Image.new("RGB", (320 + (i % 5) * 20, 240 + (i % 7) * 15), ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255))
        d = ImageDraw.Draw(img)
        d.text((18, 18), f"{cat} #{i}", fill=(255, 255, 255))
        path = OUT / f"sample_{i:04d}_{cat}.jpg"
        img.save(path, quality=86)
        rows.append(f"{path.name},{cat},{cat}|demo|sample\n")
    (OUT / "metadata.csv").write_text("file_name,category,labels\n" + "".join(rows))
    print(f"Generated {N} sample images in {OUT}")


if __name__ == "__main__":
    main()
