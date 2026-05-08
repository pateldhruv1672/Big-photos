"""
Part 4: Build UI Aggregates + Spark Stories

Reads enriched (or basic fallback) metadata, produces:
  - User gallery Parquet (HDFS + local)
  - Stories Parquet (HDFS + local) — groups by user_id + location
  - Stories summary JSON (local outputs/metadata/stories_summary.json)
  - Dashboard metrics JSON (local outputs/eda/dashboard_metrics.json)
  - Stage validation JSON (local outputs/eda/aggregation_validation.json)

Story rule: clusters of ≥ 10 photos per (user_id, location) pair (Part 4 spec).
"""

import json
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    collect_list, first, count, concat, lit, slice as spark_slice,
    min as spark_min, max as spark_max,
)

spark = (
    SparkSession.builder
    .appName("build-ui-aggregates")
    .config("spark.sql.parquet.writeLegacyFormat", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ─── Load enriched metadata ───────────────────────────────────────────────────

def try_read(path):
    try:
        df = spark.read.parquet(path)
        df.count()  # force evaluation to catch empty-path errors
        return df
    except Exception:
        return None

enriched = (
    try_read("hdfs://namenode:9000/photos/metadata/enriched")
    or try_read("/app/outputs/metadata/enriched")
    or try_read("hdfs://namenode:9000/photos/metadata/basic")
    or try_read("/app/outputs/metadata/basic")
)

if enriched is None:
    print("No metadata found — nothing to aggregate.")
    spark.stop()
    raise SystemExit(0)

total = enriched.count()
print(f"Loaded {total} enriched rows.")

# ─── User gallery ─────────────────────────────────────────────────────────────
# Per architecture diagram: Latest images, labels, thumbnails
# thumbnail_uri = same as image_uri (HDFS path to the image, rendered by frontend)

from pyspark.sql.functions import col as _col
gallery = enriched.withColumn("thumbnail_uri", _col("image_uri"))

gallery.write.mode("overwrite").parquet("hdfs://namenode:9000/photos/aggregates/user_gallery")

local_gallery = "/app/outputs/metadata/gallery"
Path(local_gallery).mkdir(parents=True, exist_ok=True)
gallery.write.mode("overwrite").parquet(local_gallery)
print(f"Gallery written to HDFS + {local_gallery}")

# ─── Search metadata aggregate ────────────────────────────────────────────────
# Lightweight table for the vector search engine:
# image_uri, labels, category, embedding_id — as shown in architecture diagram

search_cols = ["image_id", "image_uri", "category", "caption", "embedding_dim", "taken_at", "location", "user_id"]
# Include scene_labels and vlm_labels if present
for opt_col in ["vlm_labels", "scene_labels", "objects"]:
    if opt_col in enriched.columns:
        search_cols.append(opt_col)

from pyspark.sql.functions import col as spark_col
search_meta = enriched.select(*[c for c in search_cols if c in enriched.columns])
# Add embedding_id as alias for image_id — per architecture diagram
search_meta = search_meta.withColumn("embedding_id", spark_col("image_id"))
search_meta.write.mode("overwrite").parquet("hdfs://namenode:9000/photos/aggregates/search_metadata")

local_search = "/app/outputs/metadata/search_metadata"
Path(local_search).mkdir(parents=True, exist_ok=True)
search_meta.write.mode("overwrite").parquet(local_search)
search_row_count = search_meta.count()
print(f"Search metadata written to HDFS + {local_search} ({search_row_count} rows)")

# ─── Stories ─────────────────────────────────────────────────────────────────
# Part 4 spec: keep clusters with at least 10 photos per (user_id, location).

MIN_PHOTOS = 10

aggregated = (
    enriched.groupBy("user_id", "location")
    .agg(
        count("*").alias("photo_count"),
        collect_list("image_id").alias("image_ids_all"),
        first("image_uri").alias("cover_image_uri"),
        spark_min("taken_at").alias("start_date"),
        spark_max("taken_at").alias("end_date"),
    )
)

total_clusters = aggregated.count()
_max_row = aggregated.select(spark_max("photo_count").alias("_mx")).collect()
_max_val = _max_row[0]["_mx"] if _max_row else None
max_photos_in_group = int(_max_val) if _max_val is not None else 0

stories = (
    aggregated.where(f"photo_count >= {MIN_PHOTOS}")
    .withColumn("image_ids", spark_slice("image_ids_all", 1, 10))
    .drop("image_ids_all")
    .withColumn("story_id", concat(lit("story_"), "user_id", lit("_"), "location"))
    .withColumn("title", concat(lit("Remember your trip to "), "location"))
    .withColumn("summary", concat(lit("A collection of "), "photo_count", lit(" memories from "), "location"))
)

story_count = stories.count()
print(f"Generated {story_count} stories (min_photos={MIN_PHOTOS}).")

if story_count == 0:
    print(
        f"WARNING: No story clusters meet MIN_PHOTOS={MIN_PHOTOS}. "
        f"There are {total_clusters} (user_id, location) groups in the metadata; "
        f"the largest group has {max_photos_in_group} photo(s). "
        "Gather more images per location (or consolidate tag-driven locations); the threshold is fixed by spec."
    )
elif total_clusters > story_count:
    skipped = total_clusters - story_count
    print(
        f"INFO: {story_count} of {total_clusters} location groups qualify as stories "
        f"(MIN_PHOTOS={MIN_PHOTOS}); {skipped} group(s) below threshold."
    )

stories.write.mode("overwrite").parquet("hdfs://namenode:9000/photos/aggregates/final_stories")

local_stories = "/app/outputs/metadata/stories"
Path(local_stories).mkdir(parents=True, exist_ok=True)
stories.write.mode("overwrite").parquet(local_stories)
print(f"Stories written to HDFS + {local_stories}")

# Human-readable story list for grading / quick inspection (Parquet remains canonical).
stories_json_path = Path("/app/outputs/metadata/stories_summary.json")
if story_count > 0:
    slim_rows = stories.select(
        "story_id", "location", "photo_count", "image_ids", "start_date", "end_date", "title"
    ).collect()
    stories_json_path.write_text(
        json.dumps(
            {
                "min_photos_per_story": MIN_PHOTOS,
                "story_count": story_count,
                "stories": [r.asDict() for r in slim_rows],
            },
            indent=2,
            default=str,
        )
    )
else:
    stories_json_path.write_text(
        json.dumps(
            {
                "min_photos_per_story": MIN_PHOTOS,
                "story_count": 0,
                "stories": [],
                "note": "No clusters met the minimum photo count; see Spark driver logs for WARNING details.",
            },
            indent=2,
        )
    )
print(f"Stories summary JSON → {stories_json_path}")

# ─── Dashboard metrics ────────────────────────────────────────────────────────
# Spec (Part 4): Spark aggregation must output /photos/aggregates/dashboard_metrics/

from pyspark.sql.functions import avg

cat_counts = {}
if "category" in enriched.columns:
    cat_rows = enriched.groupBy("category").count().collect()
    cat_counts = {r["category"]: r["count"] for r in cat_rows}

location_counts = {}
if "location" in enriched.columns:
    loc_rows = enriched.groupBy("location").count().orderBy("count", ascending=False).limit(10).collect()
    location_counts = {r["location"]: r["count"] for r in loc_rows}

dashboard = {
    "total_images": total,
    "total_stories": story_count,
    "category_distribution": cat_counts,
    "top_locations": location_counts,
    "min_photos_per_story": MIN_PHOTOS,
    "story_location_groups_total": total_clusters,
    "max_photos_in_largest_location_group": max_photos_in_group,
}

# Write to HDFS as Parquet (required by spec)
dashboard_df = spark.createDataFrame([dashboard])
dashboard_df.write.mode("overwrite").parquet("hdfs://namenode:9000/photos/aggregates/dashboard_metrics")
print("Dashboard metrics written to HDFS: /photos/aggregates/dashboard_metrics")

# Also write locally as JSON for convenience
metrics_out = Path("/app/outputs/eda")
metrics_out.mkdir(parents=True, exist_ok=True)
(metrics_out / "dashboard_metrics.json").write_text(json.dumps(dashboard, indent=2))
print(f"Dashboard metrics JSON: {dashboard}")

validation = {
    "stage": "build_ui_aggregates",
    "min_photos_per_story": MIN_PHOTOS,
    "rows": {
        "input_metadata": total,
        "search_metadata": search_row_count,
        "qualified_stories": story_count,
    },
    "story_clusters": {
        "location_groups_total": total_clusters,
        "max_photos_in_single_group": max_photos_in_group,
    },
    "local_output_paths": {
        "gallery_parquet": local_gallery,
        "search_metadata_parquet": local_search,
        "stories_parquet": local_stories,
        "stories_summary_json": str(stories_json_path),
        "dashboard_metrics_json": str(metrics_out / "dashboard_metrics.json"),
    },
    "hdfs_aggregate_paths": {
        "user_gallery": "hdfs://namenode:9000/photos/aggregates/user_gallery",
        "search_metadata": "hdfs://namenode:9000/photos/aggregates/search_metadata",
        "final_stories": "hdfs://namenode:9000/photos/aggregates/final_stories",
        "dashboard_metrics": "hdfs://namenode:9000/photos/aggregates/dashboard_metrics",
    },
    "input_columns": enriched.columns,
}
(metrics_out / "aggregation_validation.json").write_text(json.dumps(validation, indent=2))
print(f"Wrote aggregation validation → {metrics_out / 'aggregation_validation.json'}")

spark.stop()
print("build_ui_aggregates.py complete.")
