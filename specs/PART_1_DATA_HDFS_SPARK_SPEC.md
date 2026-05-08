# Part 1: HDFS + MIRFLICKR Ingestion + Spark EDA


# Global Project Constraints

- Everything must run with Docker Compose.
- Minimize human interaction. Target command: `make all`.
- No Postgres and no Qdrant.
- HDFS is the source of truth for images and Parquet metadata.
- Metadata must be stored as Parquet in HDFS.
- Vector search must use one lecture-covered ANN method: HNSW with `hnswlib`.
- Kafka is used as the bulk upload queue.
- Spark is used for EDA and aggregation pipelines for UI-ready results.
- Ray is used for training and Ray Serve inference.
- Ollama is used for VLM labeling.
- Use mock/sample fallbacks where large external downloads or GPU models are unavailable.


## Goal
Build the batch data foundation:
1. Load MIRFLICKR or sample images.
2. Store raw images in HDFS.
3. Create `basic_metadata.parquet`.
4. Run Spark EDA.
5. Produce charts and tabular outputs for the report/UI.

## Required Services
- `namenode`
- `datanode`
- `spark-master`
- `spark-worker`

## Required Files to Implement
- `scripts/wait_for_services.sh`
- `scripts/generate_sample_dataset.py`
- `scripts/hdfs_init.sh`
- `spark_jobs/build_basic_metadata.py`
- `spark_jobs/eda.py`

## HDFS Layout
```text
/photos/raw/images/
/photos/metadata/basic/
/photos/metadata/enriched/
/photos/metadata/uploads/
/photos/aggregates/
/photos/models/
/photos/vector_index/
```

## Metadata Schema
```text
image_id: string
user_id: string
image_uri: string
file_name: string
dataset: string
file_size: long
width: int
height: int
tags: array<string>
taken_at: timestamp
location: string
created_at: timestamp
```

## Success Criteria
- `make part1` runs successfully.
- HDFS contains raw images.
- HDFS contains basic metadata Parquet.
- `outputs/eda` contains at least:
  - `tag_frequency.csv`
  - `image_dimensions.csv`
  - `dataset_summary.json`
