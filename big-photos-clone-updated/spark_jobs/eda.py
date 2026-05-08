from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, percentile_approx

HDFS_RPC = os.getenv("HDFS_NAMENODE_RPC", "hdfs://namenode:9000").rstrip("/")
BASIC_TEAM_ROOT = os.getenv("BASIC_METADATA_ROOT", "/photos/metadata/basic")
BASIC_MIR_ROOT = os.getenv("MIRFLICKR25K_BASIC_ROOT", "/photos/metadata/mirflickr25k/basic")
OUTPUT_DIR = os.getenv("EDA_OUTPUT_DIR", "/app/outputs/eda")


def hdfs_uri(path: str) -> str:
    return path if path.startswith("hdfs://") else f"{HDFS_RPC}{path if path.startswith('/') else '/' + path}"


def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(df[x].astype(str), df[y])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["team_gallery", "mirflickr25k"], default=os.getenv("DATASET_NAME", "mirflickr25k"))
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    input_root = args.input or (BASIC_TEAM_ROOT if args.dataset == "team_gallery" else BASIC_MIR_ROOT)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName(f"big-photos-eda-{args.dataset}").getOrCreate()
    df = spark.read.parquet(hdfs_uri(input_root))
    valid_df = df.filter(col("is_valid_image") == True)

    tag_freq = valid_df.select(explode(col("tags")).alias("tag")).groupBy("tag").count().orderBy(col("count").desc())
    tag_pdf = tag_freq.limit(30).toPandas()
    tag_pdf.to_csv(out_dir / "tag_frequency.csv", index=False)
    if not tag_pdf.empty:
        save_bar(tag_pdf.head(20), "tag", "count", out_dir / "tag_frequency.png", "Top Tags")

    dims_pdf = valid_df.select("image_id", "width", "height", "file_size").toPandas()
    dims_pdf.to_csv(out_dir / "image_dimensions.csv", index=False)
    if not dims_pdf.empty:
        plt.figure(figsize=(8, 6))
        plt.scatter(dims_pdf["width"], dims_pdf["height"], s=5)
        plt.xlabel("width")
        plt.ylabel("height")
        plt.title("Image Dimensions")
        plt.tight_layout()
        plt.savefig(out_dir / "resolution_scatter.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(dims_pdf["file_size"].dropna(), bins=40)
        plt.xlabel("file_size bytes")
        plt.ylabel("count")
        plt.title("File Size Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "image_size_histogram.png")
        plt.close()

    shard_pdf = df.withColumn("shard", (col("file_name").substr(1, 1))).groupBy("shard").count().orderBy("shard").toPandas()
    shard_pdf.to_csv(out_dir / "shard_counts.csv", index=False)
    if not shard_pdf.empty:
        save_bar(shard_pdf, "shard", "count", out_dir / "shard_distribution.png", "Shard/File Prefix Distribution")

    category_col = "category" if "category" in valid_df.columns else "dataset"
    cat_pdf = valid_df.groupBy(category_col).count().orderBy(col("count").desc()).toPandas()
    cat_pdf.to_csv(out_dir / "category_distribution.csv", index=False)
    if not cat_pdf.empty:
        save_bar(cat_pdf, category_col, "count", out_dir / "category_distribution.png", "Category Distribution")

    summary = {
        "dataset": args.dataset,
        "total_rows": df.count(),
        "valid_images": valid_df.count(),
        "invalid_images": df.filter(col("is_valid_image") != True).count(),
        "file_size_percentiles": [float(x) for x in valid_df.select(percentile_approx("file_size", [0.25, 0.5, 0.75, 0.95])).first()[0]],
        "output_dir": str(out_dir),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    spark.stop()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
