# Part 2: Ollama VLM Labeling + Enriched Metadata


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
Generate image captions, labels, categories, and embeddings, then write enriched Parquet to HDFS.

## Required Services
- `ollama`
- `ray-head`
- `namenode`
- `datanode`

## Required Files to Implement
- `scripts/pull_ollama_model.sh`
- `ray_jobs/vlm_enrich.py`
- `src/common/hdfs.py`
- `src/common/embeddings.py`

## Input
```text
/photos/metadata/basic/*.parquet
/photos/raw/images/*
```

## Output
```text
/photos/metadata/enriched/date=YYYY-MM-DD/*.parquet
```

## Enriched Schema
```text
image_id
user_id
image_uri
caption
vlm_labels
objects
category
embedding
embedding_dim
quality_score
processed_at
```

## Notes
- Use Ollama vision model if available.
- If unavailable, use deterministic fallback labels from existing tags/file name.
- Embeddings should be deterministic and fixed length, preferably 128 or 512 dims.
- Store embeddings in Parquet as array<float>.

## Success Criteria
- `make part2` enriches at least sample images.
- Output rows contain caption, category, labels, and embedding.
