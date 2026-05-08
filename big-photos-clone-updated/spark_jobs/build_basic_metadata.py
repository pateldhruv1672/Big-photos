from __future__ import annotations

import argparse
import io
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List

from PIL import Image, ImageOps
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.common.image_utils import (
    image_id_from_path,
    infer_fallback_labels,
    infer_fallback_location,
    infer_fallback_taken_at,
)

HDFS_RPC = os.getenv("HDFS_NAMENODE_RPC", "hdfs://namenode:9000").rstrip("/")
RAW_TEAM_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")
RAW_MIR_ROOT = os.getenv("MIRFLICKR25K_RAW_ROOT", "/photos/raw/mirflickr25k/images")
BASIC_TEAM_ROOT = os.getenv("BASIC_METADATA_ROOT", "/photos/metadata/basic")
BASIC_MIR_ROOT = os.getenv("MIRFLICKR25K_BASIC_ROOT", "/photos/metadata/mirflickr25k/basic")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")


def hdfs_uri(path: str) -> str:
    return path if path.startswith("hdfs://") else f"{HDFS_RPC}{path if path.startswith('/') else '/' + path}"


SCHEMA = StructType(
    [
        StructField("image_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("image_uri", StringType(), False),
        StructField("thumbnail_uri", StringType(), True),
        StructField("file_name", StringType(), False),
        StructField("dataset", StringType(), False),
        StructField("file_size", LongType(), True),
        StructField("width", IntegerType(), True),
        StructField("height", IntegerType(), True),
        StructField("tags", ArrayType(StringType()), True),
        StructField("taken_at", TimestampType(), True),
        StructField("location", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("is_valid_image", BooleanType(), True),
        StructField("error_message", StringType(), True),
    ]
)


def inspect_partition(rows: Iterable) -> Iterator[Dict]:
    for row in rows:
        path = row.path
        file_name = os.path.basename(path)
        lower = file_name.lower()
        image_id = image_id_from_path(path)
        tags = infer_fallback_labels(path)
        base = {
            "image_id": image_id,
            "user_id": DEFAULT_USER_ID,
            "image_uri": path,
            "thumbnail_uri": None,
            "file_name": file_name,
            "dataset": os.getenv("DATASET_NAME", "mirflickr25k"),
            "file_size": int(row.length or 0),
            "width": 0,
            "height": 0,
            "tags": tags,
            "taken_at": infer_fallback_taken_at(path).replace(tzinfo=None),
            "location": infer_fallback_location(path),
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "is_valid_image": False,
            "error_message": None,
        }
        if not lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            base["error_message"] = "not_an_image_file"
            yield base
            continue
        try:
            with Image.open(io.BytesIO(row.content)) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
            base.update({"width": int(width), "height": int(height), "is_valid_image": True})
        except Exception as exc:
            base["error_message"] = str(exc)[:200]
        yield base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["team_gallery", "mirflickr25k"], default=os.getenv("DATASET_NAME", "mirflickr25k"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    raw_root = args.input or (RAW_TEAM_ROOT if args.dataset == "team_gallery" else RAW_MIR_ROOT)
    output_root = args.output or (BASIC_TEAM_ROOT if args.dataset == "team_gallery" else BASIC_MIR_ROOT)
    os.environ["DATASET_NAME"] = args.dataset

    spark = SparkSession.builder.appName(f"big-photos-basic-metadata-{args.dataset}").getOrCreate()
    # binaryFile recursively scans images from HDFS without loading all paths on the driver.
    binary = spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(hdfs_uri(raw_root))
    rdd = binary.rdd.mapPartitions(inspect_partition)
    df = spark.createDataFrame(rdd, schema=SCHEMA)
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    out = hdfs_uri(f"{output_root.rstrip('/')}/{date_part}")
    df.write.mode("overwrite").parquet(out)
    print(f"Wrote basic metadata to {out}; rows={df.count()}")
    spark.stop()


if __name__ == "__main__":
    main()
