"""
Part 1: Build Basic Metadata

Scans local MIRFLICKR images, builds a metadata DataFrame, writes it to:
  - HDFS:  hdfs://namenode:9000/photos/metadata/basic/
  - Local: /app/outputs/metadata/basic/  (so Ray jobs can read without Spark)

Also uploads raw images to HDFS /photos/raw/images/.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from pathlib import Path
import json, os

spark = (
    SparkSession.builder
    .appName("build-basic-metadata")
    .config("spark.sql.parquet.writeLegacyFormat", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ─── Load images and tags ─────────────────────────────────────────────────────

local_dir = Path("/app/data/mirflickr/images")
tags_path  = Path("/app/data/mirflickr/tags.json")
tags = json.loads(tags_path.read_text()) if tags_path.exists() else {}

if not local_dir.exists() or not any(local_dir.glob("*.jpg")):
    print("No images found — running generate_sample_dataset.py first...")
    import subprocess
    subprocess.run(["python", "/app/scripts/generate_sample_dataset.py"], check=True)

rows = []
for p in sorted(local_dir.glob("*.jpg")):
    image_id = p.stem
    try:
        from PIL import Image
        with Image.open(p) as img:
            width, height = img.size
    except Exception:
        width, height = 256, 256

    meta = tags.get(image_id, {})
    rows.append({
        "image_id":  image_id,
        "user_id":   "demo_user",
        "image_uri": f"hdfs://namenode:9000/photos/raw/images/{p.name}",
        "file_name": p.name,
        "dataset":   meta.get("source", "MIRFLICKR_25K"),
        "file_size": int(p.stat().st_size),
        "width":     width,
        "height":    height,
        "tags":      meta.get("tags", []),
        "annotation_labels": meta.get("annotation_labels", meta.get("tags", [])),
        "taken_at":  meta.get("taken_at", "2019-07-01T10:00:00"),
        "location":  meta.get("location", "Unknown"),
    })

print(f"Scanned {len(rows)} images.")

# ─── Upload images to HDFS ────────────────────────────────────────────────────

print("Uploading images to HDFS via WebHDFS...")
import requests as _req
_uploaded = 0
for _img in sorted(local_dir.glob("*.jpg"))[:200]:
    try:
        _r = _req.put(
            f"http://namenode:9870/webhdfs/v1/photos/raw/images/{_img.name}?op=CREATE&user.name=root&overwrite=true",
            allow_redirects=False, timeout=10,
        )
        _loc = _r.headers.get("Location")
        if _loc:
            _req.put(_loc, data=_img.read_bytes(),
                     headers={"Content-Type": "application/octet-stream"}, timeout=30)
            _uploaded += 1
    except Exception:
        pass
print(f"Uploaded {_uploaded} images to HDFS via WebHDFS.")

# ─── Write metadata Parquet ───────────────────────────────────────────────────

df = spark.createDataFrame(rows).withColumn("created_at", current_timestamp())

# Write to HDFS
hdfs_out = "hdfs://namenode:9000/photos/metadata/basic"
df.write.mode("overwrite").parquet(hdfs_out)
print(f"Wrote {df.count()} rows to HDFS: {hdfs_out}")

# Also write locally so Ray jobs (which don't have Spark) can read it
local_out = "/app/outputs/metadata/basic"
Path(local_out).mkdir(parents=True, exist_ok=True)
df.write.mode("overwrite").parquet(local_out)
print(f"Wrote local copy: {local_out}")

spark.stop()
print("build_basic_metadata.py complete.")
