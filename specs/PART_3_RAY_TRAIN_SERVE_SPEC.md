# Part 3: Ray Training + Ray Serve Inference


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
Train an image classifier using Ray and deploy it with Ray Serve.

## Required Services
- `ray-head`
- `ray-worker`
- `namenode`
- `datanode`

## Required Files to Implement
- `ray_jobs/train_classifier.py`
- `serve/ray_serve_app.py`
- `scripts/start_ray_serve.sh`
- `tests/test_ray_serve.py`

## Input
```text
/photos/metadata/enriched/**/*.parquet
```

## Output
```text
/photos/models/image_classifier/model.pkl
```

## Ray Serve API
Endpoint:
```text
POST /predict
```

Input:
```json
{
  "image_id": "sample_001",
  "image_uri": "hdfs://namenode:9000/photos/raw/images/sample_001.jpg"
}
```

Output:
```json
{
  "image_id": "sample_001",
  "predicted_category": "travel",
  "confidence": 0.87,
  "labels": ["city", "bridge", "travel"]
}
```

## Success Criteria
- `make part3` trains model and starts Ray Serve.
- `curl localhost:8000/predict` returns JSON.
