# Part 4: Web App + Kafka Upload + HNSW Search + Spark Stories


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
Build the online application:
1. User bulk uploads images.
2. Backend writes images to HDFS and publishes Kafka event.
3. Consumer reads Kafka, calls Ray Serve, generates embedding, appends Parquet metadata.
4. HNSW index is updated using `hnswlib`.
5. Spark aggregation creates story and UI-ready Parquet.
6. Frontend shows gallery, search, and stories.

## Required Services
- `backend`
- `frontend`
- `kafka`
- `upload-consumer`
- `spark-master`
- `spark-worker`
- `ray-head`
- `namenode`
- `datanode`

## Required Files to Implement
- `backend/main.py`
- `consumer/upload_consumer.py`
- `src/search/hnsw_index.py`
- `spark_jobs/build_ui_aggregates.py`
- `frontend/src/App.jsx`

## Kafka Topics
```text
image_uploaded
image_labeled
processing_failed
```

## HNSW Requirements
- Use `hnswlib`.
- Persist index to:
```text
/photos/vector_index/hnsw_index.bin
/photos/vector_index/id_map.json
```
- Use cosine similarity.

## Spark Aggregation Outputs
```text
/photos/aggregates/user_gallery/
/photos/aggregates/final_stories/
/photos/aggregates/dashboard_metrics/
```

## Story Rule
- Group by `user_id`, `location`, and time window.
- Keep clusters with at least 10 photos.
- Select top 10.
- Generate:
  - story_id
  - title
  - summary
  - image_ids
  - cover_image_uri

## Success Criteria
- `make part4` starts backend/frontend/consumer.
- Upload sample photos.
- UI displays gallery, search results, and stories.
