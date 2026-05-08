from __future__ import annotations

import argparse
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    array_distinct,
    col,
    collect_list,
    concat_ws,
    count,
    date_trunc,
    flatten,
    first,
    lit,
    max as spark_max,
    min as spark_min,
    slice,
    row_number,
)

HDFS_RPC = os.getenv("HDFS_NAMENODE_RPC", "hdfs://namenode:9000").rstrip("/")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
UPLOAD_METADATA_ROOT = os.getenv("UPLOAD_METADATA_ROOT", "/photos/metadata/uploads")
AGG_ROOT = os.getenv("AGGREGATES_ROOT", "/photos/aggregates")


def hdfs_uri(path: str) -> str:
    return path if path.startswith("hdfs://") else f"{HDFS_RPC}{path if path.startswith('/') else '/' + path}"


def read_metadata_root(spark: SparkSession, root: str):
    base = hdfs_uri(root).rstrip("/")
    candidates = [
        f"{base}/**/*_shard_metadata.parquet",
        f"{base}/**/*imported_metadata.parquet",
        f"{base}/**/*ui_active*.parquet",
        f"{base}/**/*.parquet",
    ]
    for p in candidates:
        try:
            df = spark.read.option("recursiveFileLookup", "true").parquet(p)
            if df is not None and len(df.columns) > 0:
                return df
        except Exception:
            continue
    raise RuntimeError(f"No readable parquet files under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=UI_ACTIVE_METADATA_ROOT)
    args = parser.parse_args()
    spark = SparkSession.builder.appName("big-photos-ui-aggregates").getOrCreate()
    frames = []
    for root in [args.input, UPLOAD_METADATA_ROOT]:
        try:
            frames.append(read_metadata_root(spark, root))
        except Exception as exc:
            print(f"Skipping {root}: {exc}")
    if not frames:
        raise SystemExit("No UI metadata available for aggregates")
    df = frames[0]
    for f in frames[1:]:
        df = df.unionByName(f, allowMissingColumns=True)
    if "deleted" in df.columns:
        df = df.filter((col("deleted") == False) | col("deleted").isNull())
    if "dataset" in df.columns:
        df = df.filter(col("dataset").isin("team_gallery", "uploads"))
    df = df.filter(col("image_id").isNotNull() & col("image_uri").isNotNull())

    # Keep newest record per image_id for deterministic cards/stories.
    if "updated_at" in df.columns:
        w = Window.partitionBy("image_id").orderBy(col("updated_at").desc_nulls_last())
        df = df.withColumn("_rn", row_number().over(w)).filter(col("_rn") == 1).drop("_rn")
    else:
        df = df.dropDuplicates(["image_id"])
    gallery = df.select("image_id", "user_id", "image_uri", "thumbnail_uri", "caption", "category", "location", "taken_at")
    gallery.write.mode("overwrite").parquet(hdfs_uri(f"{AGG_ROOT}/user_gallery"))

    base = df.withColumn("time_window", date_trunc("month", col("taken_at")))
    grouped = base.groupBy("user_id", "location", "time_window").agg(
        count("*").alias("photo_count"),
        collect_list("image_id").alias("all_image_ids"),
        first("thumbnail_uri", ignorenulls=True).alias("cover_image_uri"),
        first("category", ignorenulls=True).alias("category"),
        collect_list("labels").alias("all_labels"),
        spark_min("taken_at").alias("time_window_start"),
        spark_max("taken_at").alias("time_window_end"),
    )
    stories = grouped.filter(col("photo_count") >= 10).withColumn("image_ids", slice(col("all_image_ids"), 1, 10))
    stories = stories.withColumn("story_id", concat_ws("_", lit("story"), col("user_id"), col("location"), col("time_window").cast("string")))
    stories = stories.withColumn("title", concat_ws(" ", col("location"), lit("memories")))
    stories = stories.withColumn("summary", concat_ws(" ", lit("A collection of"), col("photo_count").cast("string"), lit("photos from"), col("location")))
    stories = stories.withColumn("top_labels", slice(array_distinct(flatten(col("all_labels"))), 1, 10))
    stories.select(
        "story_id",
        "user_id",
        "location",
        "time_window_start",
        "time_window_end",
        "title",
        "summary",
        "image_ids",
        "cover_image_uri",
        "photo_count",
        "top_labels",
        "category",
    ).write.mode("overwrite").parquet(hdfs_uri(f"{AGG_ROOT}/final_stories"))

    metrics = df.groupBy("category").count().orderBy(col("count").desc())
    metrics.write.mode("overwrite").parquet(hdfs_uri(f"{AGG_ROOT}/dashboard_metrics"))
    print("Wrote UI aggregates")
    spark.stop()


if __name__ == "__main__":
    main()
