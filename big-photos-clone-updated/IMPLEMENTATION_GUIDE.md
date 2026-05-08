# Big Photos Clone: Implementation Guide

Date: 2026-05-08

This repository implements a Google Photos clone aligned with the project constraints:

- Docker Compose only.
- Minimal human interaction through `make` commands.
- No Postgres and no Qdrant.
- HDFS is the source of truth for raw images, thumbnails, model artifacts, vector index artifacts, and Parquet metadata.
- Metadata is stored as Parquet in HDFS.
- Vector search uses HNSW with `hnswlib`.
- Kafka is used as the bulk upload queue.
- Spark is used for EDA and UI/story aggregation.
- Ray is used for training and Ray Serve inference.
- Ollama is used for VLM labeling when available; deterministic fallbacks keep the pipeline runnable without GPU/model downloads.

---

## 1. Current design decision

The project uses two data paths.

### UI/search path

The UI uses your existing indexed data and your existing HDFS image collection:

```text
/photos/raw/team_gallery/images
```

Metadata files that currently live on the normal filesystem are imported, normalized, written to HDFS Parquet, and indexed with HNSW.

### ML/EDA path

MIRFLICKR 25K is used for:

- Spark basic metadata and EDA.
- Ollama VLM labeling, or fallback labeling when Ollama is unavailable.
- Ray training.
- Ray Serve inference model.

This avoids trying to reprocess all 100k UI images during a live demo.

---

## 2. Implemented components

### Infrastructure

File: `docker-compose.yml`

Services implemented:

```text
namenode
datanode
spark-master
spark-worker
kafka
ollama
ray-head
ray-worker
backend
upload-consumer
frontend
```

All application containers mount:

```text
./data    -> /app/data
./outputs -> /app/outputs
./state   -> /app/state
```

### HDFS utilities

File: `src/common/hdfs.py`

Implemented:

- WebHDFS path normalization.
- HDFS `mkdirs`, `exists`, `list_status`, recursive `walk`.
- Binary read/write.
- Streaming byte iterator for image serving.
- Local file upload/download.
- JSON read/write.
- Parquet dataset read/write helpers.

### Metadata normalization

File: `src/common/metadata_normalizer.py`

Implemented:

- Flexible local metadata import from inconsistent schemas.
- Column aliases for image IDs, HDFS paths, captions, labels, vectors, dimensions, and timestamps.
- Deterministic fallback labels/category/caption.
- Deterministic embedding generation when no embedding is present.
- Standard UI schema:

```text
image_id
user_id
image_uri
thumbnail_uri
file_name
dataset
caption
labels
vlm_labels
objects
category
embedding
embedding_dim
width
height
file_size
taken_at
location
deleted
created_at
updated_at
```

### Existing metadata import

File: `scripts/import_existing_metadata.py`

Implemented:

- Reads `csv`, `json`, `jsonl`, and `parquet` from:

```text
/app/data/existing_metadata
```

- If no local metadata exists, scans HDFS raw images under:

```text
/photos/raw/team_gallery/images
```

- Writes:

```text
/photos/metadata/imported/date=YYYY-MM-DD/imported_metadata.parquet
/photos/metadata/ui_active/date=YYYY-MM-DD/ui_active.parquet
```

### Thumbnail generation

File: `scripts/generate_missing_thumbnails.py`

Implemented:

- Reads active metadata.
- Pulls full image bytes from HDFS.
- Generates 256px thumbnails.
- Writes thumbnails to:

```text
/photos/thumbnails/team_gallery/
```

- Rewrites active metadata snapshot with `thumbnail_uri`.

### HNSW vector index

File: `src/search/hnsw_index.py`

Implemented:

- Builds cosine HNSW index using `hnswlib`.
- Uses integer labels internally, as required by `hnswlib`.
- Persists artifacts to HDFS:

```text
/photos/vector_index/current/hnsw_index.bin
/photos/vector_index/current/id_map.json
/photos/vector_index/current/manifest.json
/photos/vector_index/current/item_meta.json
```

`id_map.json` format:

```json
{
  "0": "image_id_1",
  "1": "image_id_2"
}
```

Implemented index rules:

- Drop deleted rows.
- Drop rows without `image_id`.
- Drop rows without valid embeddings.
- Deduplicate by `image_id`.
- Enforce fixed embedding dimension.
- Use cosine similarity.

### Existing index sync

File: `scripts/sync_existing_index.py`

Implemented:

- Copies local existing index artifacts into HDFS when you already have a valid index.
- Validates that `id_map.json` has integer-like keys.
- Writes a manifest.

Recommended default is still to rebuild from metadata using:

```bash
make build-index
```

### UI index validation

File: `scripts/validate_ui_index.py`

Implemented:

- Checks active metadata row count.
- Checks HNSW vector count.
- Checks index/id-map/manifest presence in HDFS.
- Verifies indexed image IDs exist in metadata.

### Backend

File: `backend/main.py`

Implemented API:

```text
GET  /health
POST /api/refresh
GET  /api/gallery
GET  /api/thumb/{image_id}
GET  /api/image/{image_id}
POST /api/search
POST /api/upload
POST /api/delete
GET  /api/stories
```

Behavior:

- Loads UI metadata from HDFS Parquet into a demo cache.
- Loads HNSW index from HDFS.
- Streams thumbnails and full images from HDFS through FastAPI.
- Searches HNSW using deterministic text embeddings.
- Writes uploaded images to HDFS and publishes Kafka events.
- Writes delete tombstones to HDFS Parquet.
- Rebuilds HNSW after deletes for consistency.

### Kafka upload consumer

File: `consumer/upload_consumer.py`

Implemented:

- Consumes `image_uploaded` events.
- Reads uploaded image from HDFS.
- Creates thumbnail.
- Calls Ray Serve `/predict`.
- Falls back to deterministic labels if Ray Serve is unavailable.
- Generates embedding.
- Appends upload metadata Parquet to HDFS.
- Updates HNSW artifacts in HDFS.
- Publishes `image_labeled` or `processing_failed` events.

### Spark basic metadata

File: `spark_jobs/build_basic_metadata.py`

Implemented:

- Reads images from HDFS using Spark `binaryFile`.
- Supports `--dataset mirflickr25k` and `--dataset team_gallery`.
- Extracts file size, width, height, valid-image flag, fallback tags, location, timestamp.
- Writes Parquet to:

```text
/photos/metadata/mirflickr25k/basic/date=YYYY-MM-DD
```

or the team-gallery basic metadata root.

### Spark EDA

File: `spark_jobs/eda.py`

Implemented outputs:

```text
outputs/eda/tag_frequency.csv
outputs/eda/image_dimensions.csv
outputs/eda/dataset_summary.json
outputs/eda/shard_counts.csv
outputs/eda/tag_frequency.png
outputs/eda/resolution_scatter.png
outputs/eda/image_size_histogram.png
outputs/eda/shard_distribution.png
```

### Spark UI aggregates and stories

File: `spark_jobs/build_ui_aggregates.py`

Implemented outputs:

```text
/photos/aggregates/user_gallery/
/photos/aggregates/final_stories/
/photos/aggregates/dashboard_metrics/
```

Story rule:

- Group by `user_id`, `location`, and monthly time window.
- Keep groups with at least 10 photos.
- Select top 10 image IDs.
- Generate story ID, title, summary, cover image URI, and photo count.

### Ollama VLM enrichment

File: `ray_jobs/vlm_enrich.py`

Implemented:

- Reads MIRFLICKR 25K basic metadata.
- Tries Ollama vision model with image bytes.
- If unavailable, uses deterministic fallback captions, labels, objects, and categories.
- Runs enrichment in Ray remote tasks when Ray is available.
- Writes enriched Parquet to:

```text
/photos/metadata/mirflickr25k/enriched/date=YYYY-MM-DD
```

### Ray training

File: `ray_jobs/train_classifier.py`

Implemented:

- Reads MIRFLICKR 25K enriched metadata.
- Converts embeddings/dimensions/file size into feature vectors using Ray remote tasks.
- Trains a scikit-learn RandomForest classifier.
- Writes model to HDFS:

```text
/photos/models/image_classifier/model.pkl
```

- Writes metrics to HDFS:

```text
/photos/models/image_classifier/metrics.json
```

### Ray Serve

File: `serve/ray_serve_app.py`

Implemented:

- Loads HDFS model artifact if present.
- Exposes:

```text
POST /predict
```

- Returns:

```json
{
  "image_id": "sample_001",
  "predicted_category": "travel",
  "confidence": 0.87,
  "labels": ["city", "bridge", "travel"]
}
```

- Uses deterministic fallback when the model is missing.

### Frontend

Files:

```text
frontend/src/App.jsx
frontend/src/styles.css
```

Implemented:

- Gallery.
- Search.
- Stories.
- Bulk upload.
- Bulk delete with checkboxes.
- Full image modal.
- Thumbnails loaded from `/api/thumb/{image_id}`.
- Full images loaded from `/api/image/{image_id}`.

---

## 3. HDFS layout

The initialization script creates:

```text
/photos/raw/images/
/photos/raw/team_gallery/images/
/photos/raw/mirflickr25k/images/
/photos/raw/uploads/
/photos/thumbnails/team_gallery/
/photos/thumbnails/mirflickr25k/
/photos/thumbnails/uploads/
/photos/metadata/basic/
/photos/metadata/imported/
/photos/metadata/ui_active/
/photos/metadata/mirflickr25k/basic/
/photos/metadata/mirflickr25k/enriched/
/photos/metadata/uploads/
/photos/metadata/deletes/
/photos/aggregates/user_gallery/
/photos/aggregates/final_stories/
/photos/aggregates/dashboard_metrics/
/photos/models/image_classifier/
/photos/vector_index/current/
/photos/events/uploads/
/photos/events/deletes/
```

---

## 4. How to run on your existing 100k HDFS dataset

### Step 1: Configure `.env`

```bash
cp .env.example .env
```

Set this to match your verified HDFS path:

```text
RAW_UI_IMAGE_ROOT=/photos/raw/team_gallery/images
```

Place your local metadata files here:

```text
./data/existing_metadata/
```

Supported formats:

```text
.csv
.json
.jsonl
.parquet
```

If the metadata has embeddings, the HNSW index will be built from them. If it does not, deterministic embeddings are generated from caption/labels/category.

### Step 2: Start services

```bash
make up
```

### Step 3: Import and index existing metadata

```bash
make index-existing
```

This runs:

```bash
python scripts/import_existing_metadata.py
python scripts/generate_missing_thumbnails.py
python scripts/build_hnsw_from_metadata.py
python scripts/validate_ui_index.py
```

### Step 4: Start UI

```bash
make ui
```

Open:

```text
http://localhost:5173
```

---

## 5. How to run without external data

For a local smoke demo:

```bash
cp .env.example .env
make up
make sample-data
make index-existing
make ui
```

This creates sample images, uploads them to HDFS, writes sample metadata, builds thumbnails, builds HNSW, and starts the UI.

---

## 6. MIRFLICKR 25K ML/EDA pipeline

Put MIRFLICKR 25K images in HDFS under:

```text
/photos/raw/mirflickr25k/images
```

Then run:

```bash
make ml-mirflickr25k
```

This runs:

```bash
make part1   # Spark metadata + EDA
make part2   # Ollama/fallback VLM enrichment
make part3   # Ray training + Ray Serve start
```

EDA files are written to:

```text
outputs/eda/
```

Model artifacts are written to:

```text
/photos/models/image_classifier/
```

---

## 7. UI image-serving behavior

The browser never calls HDFS directly.

Gallery thumbnail:

```text
frontend -> GET /api/thumb/{image_id} -> backend -> HDFS thumbnail bytes
```

Full image click:

```text
frontend -> GET /api/image/{image_id} -> backend -> HDFS full image bytes
```

This keeps HDFS internal and makes the UI simpler.

---

## 8. HNSW behavior

Build index:

```bash
make build-index
```

Validate:

```bash
make validate-index
```

Search endpoint:

```bash
curl -X POST http://localhost:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"beach travel outdoor","top_k":10}'
```

Artifacts in HDFS:

```text
/photos/vector_index/current/hnsw_index.bin
/photos/vector_index/current/id_map.json
/photos/vector_index/current/manifest.json
/photos/vector_index/current/item_meta.json
```

---

## 9. Upload and delete behavior

### Upload

```text
Frontend bulk upload
  -> backend writes raw image to HDFS
  -> backend publishes Kafka image_uploaded event
  -> upload-consumer reads event
  -> consumer creates thumbnail
  -> consumer calls Ray Serve /predict
  -> consumer writes upload metadata Parquet
  -> consumer updates HNSW index in HDFS
```

### Delete

```text
Frontend bulk delete
  -> backend writes delete tombstones to /photos/metadata/deletes
  -> backend rebuilds HNSW excluding deleted image IDs
  -> UI no longer shows deleted images
```

Physical HDFS image deletion is intentionally not performed during the demo.

---

## 10. Continue/Cursor development guide

Open the repository in Cursor, then use Continue with these prompts:

### To inspect the architecture

```text
Read IMPLEMENTATION_GUIDE.md, README.md, docker-compose.yml, Makefile, backend/main.py, src/search/hnsw_index.py, and scripts/import_existing_metadata.py. Summarize how UI metadata is imported and how HNSW search works without Postgres/Qdrant.
```

### To connect your real metadata

```text
Inspect my files under data/existing_metadata. Update src/common/metadata_normalizer.py only if any field names are not mapped. Do not change the HDFS schema unless necessary.
```

### To improve image embeddings later

```text
Replace deterministic embeddings in src/common/embeddings.py with CLIP image/text embeddings while preserving the public function names text_embedding and record_embedding and the fixed EMBEDDING_DIM contract.
```

### To improve Ray training later

```text
Modify ray_jobs/train_classifier.py and serve/ray_serve_app.py so Ray trains on actual image pixels or CLIP embeddings while preserving the HDFS model path and /predict JSON contract.
```

### To debug UI image loading

```text
Trace GET /api/gallery, /api/thumb/{image_id}, and /api/image/{image_id}. Verify metadata_by_id has image_uri and thumbnail_uri, then verify src/common/hdfs.py can read those paths through WebHDFS.
```

---

## 11. Known limitations

- The backend caches metadata in memory for demo speed. This is acceptable for 100k rows, but a production system would use a serving database or a partitioned metadata API.
- `hnswlib` index updates are single-writer. The project assumes one upload consumer for correctness.
- Delete behavior uses tombstones and HNSW rebuilds. Physical deletion from HDFS is not done in the demo.
- Ollama and Ray Serve have deterministic fallbacks so the pipeline runs without GPU/model availability.
- If local metadata does not include real embeddings, HNSW uses deterministic text embeddings from captions/labels/categories.

---

## 12. Final presentation statement

Use this wording:

> The UI uses the already indexed 100k-image HDFS collection. We import existing metadata from the local filesystem, normalize it into HDFS Parquet, generate thumbnails, and build a persisted HNSW index with hnswlib. The browser requests thumbnails and full images through the FastAPI backend, which streams them from HDFS. Separately, we use MIRFLICKR 25K to demonstrate Spark EDA, Ollama VLM labeling, Ray training, and Ray Serve inference. This keeps HDFS as the source of truth and avoids Postgres or Qdrant.
