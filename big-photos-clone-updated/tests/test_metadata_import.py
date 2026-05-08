from __future__ import annotations

import sys

import pandas as pd

from src.common.metadata_normalizer import normalize_dataframe


def main() -> int:
    df = pd.DataFrame(
        [
            {
                "id": "img_001",
                "filename": "img_001.jpg",
                "path": "/photos/raw/team_gallery/images/0/img_001.jpg",
                "tags": ["beach", "sunset"],
                "user": "team_gallery",
            },
            {
                "id": "img_002",
                "filename": "img_002.jpg",
                "path": "/photos/raw/team_gallery/images/0/img_002.jpg",
                "tags": "city,night",
            },
        ]
    )
    out = normalize_dataframe(df, dataset="team_gallery", raw_root="/photos/raw/team_gallery/images")
    assert len(out) == 2
    assert "image_id" in out.columns
    assert "embedding" in out.columns
    assert out["image_id"].nunique() == 2
    assert all(isinstance(x, list) for x in out["labels"].tolist())
    print("metadata normalization smoke test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
