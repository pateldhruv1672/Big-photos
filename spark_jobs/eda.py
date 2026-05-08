"""
Part 1: Spark EDA Job

Reads basic_metadata Parquet from HDFS and produces EDA reports:
  outputs/eda/tag_frequency.csv        — tag frequency ranking
  outputs/eda/label_distribution.csv   — location/label distribution
  outputs/eda/image_dimensions.csv     — width, height, file_size per image
  outputs/eda/dataset_summary.json     — aggregate stats
  outputs/eda/tag_frequency.png        — bar chart: top 20 tags
  outputs/eda/label_distribution.png   — bar chart: label distribution
  outputs/eda/image_size_histogram.png — histogram: image width distribution

Matches architecture diagram: "Output: EDA Reports (Charts, CSV, Insights)"
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, count, avg, min as spark_min, max as spark_max
from pathlib import Path
import json

spark = SparkSession.builder.appName("spark-eda").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

out = Path("/app/outputs/eda")
out.mkdir(parents=True, exist_ok=True)

# ─── Load basic metadata ──────────────────────────────────────────────────────

df = spark.read.parquet("hdfs://namenode:9000/photos/metadata/basic")
total = df.count()
print(f"EDA: loaded {total} images.")

# ─── 1. Tag frequency CSV ─────────────────────────────────────────────────────

tag_freq = (
    df.select(explode("tags").alias("tag"))
    .groupBy("tag")
    .count()
    .orderBy("count", ascending=False)
)
tag_freq_pd = tag_freq.toPandas()
tag_freq_pd.to_csv(out / "tag_frequency.csv", index=False)
print(f"tag_frequency.csv — {len(tag_freq_pd)} unique tags")

# ─── 2. Label distribution CSV ────────────────────────────────────────────────
# Uses location as the label proxy at basic-metadata stage

location_dist = (
    df.groupBy("location")
    .count()
    .orderBy("count", ascending=False)
)
loc_pd = location_dist.toPandas().rename(columns={"location": "label", "count": "count"})
loc_pd.to_csv(out / "label_distribution.csv", index=False)
print(f"label_distribution.csv — {len(loc_pd)} labels")

# ─── 3. Image dimensions CSV ──────────────────────────────────────────────────

dims_pd = df.select("image_id", "width", "height", "file_size").toPandas()
dims_pd.to_csv(out / "image_dimensions.csv", index=False)
print(f"image_dimensions.csv — {len(dims_pd)} rows")

# ─── 4. Dataset summary JSON ──────────────────────────────────────────────────

agg = df.select(
    avg("width").alias("avg_width"),
    avg("height").alias("avg_height"),
    avg("file_size").alias("avg_file_size"),
    spark_min("width").alias("min_width"),
    spark_max("width").alias("max_width"),
    spark_min("height").alias("min_height"),
    spark_max("height").alias("max_height"),
).first()

summary = {
    "num_images": total,
    "avg_width":  round(float(agg["avg_width"] or 0), 1),
    "avg_height": round(float(agg["avg_height"] or 0), 1),
    "avg_file_size_bytes": round(float(agg["avg_file_size"] or 0), 1),
    "min_width":  int(agg["min_width"] or 0),
    "max_width":  int(agg["max_width"] or 0),
    "min_height": int(agg["min_height"] or 0),
    "max_height": int(agg["max_height"] or 0),
    "unique_tags": int(len(tag_freq_pd)),
    "unique_labels": int(len(loc_pd)),
    "top_5_tags": tag_freq_pd.head(5)[["tag", "count"]].to_dict("records"),
}
(out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
print(f"dataset_summary.json: {summary}")

# ─── 5. Charts (PNG) ──────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed inside Docker
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    PRIMARY = "#1d4e89"
    ACCENT  = "#2d6a4f"

    # Chart 1: Top 20 tag frequency (horizontal bar)
    top20 = tag_freq_pd.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top20["tag"][::-1], top20["count"][::-1], color=PRIMARY)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Top 20 Tags — MIRFLICKR Sample", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out / "tag_frequency.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("tag_frequency.png saved.")

    # Chart 2: Label / location distribution (vertical bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(loc_pd["label"], loc_pd["count"], color=ACCENT)
    ax.set_xlabel("Label / Location", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "label_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("label_distribution.png saved.")

    # Chart 3: Image width histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dims_pd["width"].dropna(), bins=20, color=PRIMARY, edgecolor="white")
    ax.set_xlabel("Width (px)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Image Width Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "image_size_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("image_size_histogram.png saved.")

except ImportError:
    print("matplotlib not installed — skipping chart generation (install with: pip install matplotlib)")

spark.stop()
print("EDA complete. All outputs in outputs/eda/")
