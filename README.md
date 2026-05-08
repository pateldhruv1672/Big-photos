# Distributed Photo Intelligence Platform

A production-grade, end-to-end distributed ML pipeline for intelligent photo management — **11 Docker services** spanning distributed storage, event streaming, batch analytics, deep learning inference, vector search, and a real-time web dashboard.

> **MobileNet V3 Small** fine-tuned on real **MIRFLICKR-25K** images with multi-label classification, served via **Ray Serve**, indexed with **HNSW** vector search, orchestrated through **Kafka** event streaming, stored on **HDFS**, and analyzed with **Apache Spark**.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Pipeline Overview](#pipeline-overview)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Services & Ports](#services--ports)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Real-Time Upload Pipeline](#real-time-upload-pipeline)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Kafka Demo](#kafka-demo)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION LAYER                                 │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │                     React + Vite Frontend (:3000)                        │   │
│   │          Gallery  │  Semantic Search  │  Stories  │  Upload              │   │
│   └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                  │ REST API                                      │
│   ┌──────────────────────────────▼───────────────────────────────────────────┐   │
│   │                     FastAPI Backend (:8080)                              │   │
│   │    /gallery  /search  /stories  /upload  /metrics  /predict  /image     │   │
│   └───────┬──────────┬──────────┬──────────┬────────────────────────────────┘   │
│           │          │          │          │                                      │
└───────────┼──────────┼──────────┼──────────┼──────────────────────────────────────┘
            │          │          │          │
┌───────────┼──────────┼──────────┼──────────┼──────────────────────────────────────┐
│           │   DATA & ML LAYER   │          │                                      │
│           │          │          │          │                                      │
│   ┌───────▼────┐ ┌───▼──────┐ ┌▼────────┐ ┌▼──────────────────┐                 │
│   │   HNSW     │ │ Enriched │ │  Spark   │ │      Kafka         │                 │
│   │  Index     │ │ Metadata │ │ Stories  │ │   (KRaft mode)     │                 │
│   │ (hnswlib)  │ │(Parquet) │ │(Parquet) │ │ image_uploaded     │                 │
│   │  Vector    │ │          │ │          │ │ image_labeled      │                 │
│   │  Search    │ │          │ │          │ │ processing_failed  │                 │
│   └────────────┘ └──────────┘ └──────────┘ └────────┬───────────┘                 │
│                                                      │                            │
│                                            ┌─────────▼──────────┐                 │
│                                            │  Upload Consumer    │                 │
│                                            │  (Long-lived svc)   │                 │
│                                            │                     │                 │
│                                            │  1. HDFS upload     │                 │
│                                            │  2. MobileNet CNN   │                 │
│                                            │  3. Embedding gen   │                 │
│                                            │  4. HNSW update     │                 │
│                                            │  5. Parquet write   │                 │
│                                            │  6. Kafka publish   │                 │
│                                            └──┬──────┬───────┬──┘                 │
│                                               │      │       │                    │
│   ┌───────────────────────┐  ┌────────────────▼┐ ┌───▼─────┐ │                    │
│   │    Spark Cluster       │  │  Ray Serve      │ │  Ollama │ │                    │
│   │  Master + Worker       │  │  MobileNet V3   │ │  LLaVA  │ │                    │
│   │                        │  │  Small (:8000)  │ │  VLM    │ │                    │
│   │  • EDA analytics       │  │                 │ │(:11434) │ │                    │
│   │  • Basic metadata      │  │  24-class       │ │         │ │                    │
│   │  • Story generation    │  │  multi-label    │ └─────────┘ │                    │
│   │  • UI aggregates       │  │  classifier     │             │                    │
│   └───────────┬────────────┘  └─────────────────┘             │                    │
│               │                        ▲                      │                    │
│               │               ┌────────┴────────┐             │                    │
│               │               │  Ray Cluster     │             │                    │
│               │               │  Head + Worker   │             │                    │
│               │               │  Training +      │             │                    │
│               │               │  Inference       │             │                    │
│               │               └─────────────────┘             │                    │
│               │                                               │                    │
│   ┌───────────▼───────────────────────────────────────────────▼─────────────┐     │
│   │                      HDFS Cluster (Hadoop 3.2.1)                        │     │
│   │                   NameNode (:9870) + DataNode                           │     │
│   │                                                                         │     │
│   │   /photos/raw/images/          ← Original MIRFLICKR images              │     │
│   │   /photos/metadata/basic/      ← Spark-generated basic metadata         │     │
│   │   /photos/metadata/enriched/   ← VLM-enriched metadata + embeddings     │     │
│   │   /photos/metadata/uploads/    ← User upload metadata                   │     │
│   │   /photos/uploads/{user}/      ← User-uploaded images (date-partitioned)│     │
│   │   /photos/models/              ← Trained model artifacts                │     │
│   │   /photos/aggregates/          ← Spark stories & UI aggregates          │     │
│   └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
MIRFLICKR-25K Images ──► HDFS ──► Spark EDA ──► Basic Metadata (Parquet)
                                                        │
                                                        ▼
                                               Ollama VLM Enrichment
                                              (captions, labels, categories)
                                                        │
                                                        ▼
                                             sentence-transformers Embeddings
                                              (all-MiniLM-L6-v2, 128-dim)
                                                        │
                                                        ▼
                                               HNSW Vector Index (hnswlib)
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                    MobileNet V3    Ray Serve      React UI
                                     Training       Inference    (Gallery, Search,
                                    (8 epochs)      (:8000)      Stories, Upload)
```

---

## Pipeline Overview

The platform is organized into **5 sequential parts**, each building on the previous:

### Part 1 — Data Ingestion (HDFS + Spark EDA)

Extracts **100 real photographs** from the MIRFLICKR-25K dataset with 24 annotation categories, uploads them to HDFS, and runs exploratory data analysis.

- Downloads and processes real MIRFLICKR-25K images (500x333 RGB JPEGs)
- Parses 24 annotation label files per image (animals, sky, water, people, etc.)
- Uploads images to **HDFS** via the **WebHDFS REST API**
- **Apache Spark** (distributed across master + worker) builds basic metadata as Parquet:
  - `tag_frequency.csv` — frequency of all 23+ unique annotation tags
  - `image_dimensions.csv` — width, height, file size per image
  - `dataset_summary.json` — aggregate statistics
  - Visualization charts (histograms, bar plots) saved to `outputs/eda/`

### Part 2 — VLM Enrichment + Vector Index

Enriches each image with captions, labels, categories, and vector embeddings for semantic search.

- Attempts **Ollama LLaVA** vision model for image captioning and object detection
- Falls back to tag-based enrichment using MIRFLICKR annotations when VLM is unavailable
- Generates **128-dimensional embeddings** via `sentence-transformers` (all-MiniLM-L6-v2)
- Assigns display categories using a **weighted annotation scoring** system:
  - 7 categories: `people`, `nature`, `city`, `travel`, `food`, `event`, `art`
  - Higher-weight multipliers for more specific categories (food ×3, nature ×1)
- Builds **HNSW vector index** (hnswlib) with cosine similarity for semantic search

### Part 3 — Model Training + Ray Serve Deployment

Fine-tunes a CNN on the real images and deploys it as a serving endpoint.

- Fine-tunes **MobileNet V3 Small** (pretrained on ImageNet) on 100 MIRFLICKR images
- **Multi-label classification** with `BCEWithLogitsLoss` (sigmoid per class, 24 classes)
- 80/20 train/val split, **8 epochs**, image augmentation (random flip, color jitter)
- Freezes early convolutional layers, fine-tunes classifier head + last conv blocks
- Deployed via **Ray Serve** with `@serve.deployment` and `@serve.ingress(app)` decorators
- Model artifact saved to HDFS at `/photos/models/image_classifier/`

### Part 4 — Spark Aggregates + Story Generation

Generates higher-level content for the frontend dashboard.

- **Apache Spark** groups photos by location to generate story cards
- Produces gallery metadata, search metadata, dashboard metrics
- All aggregates written to both HDFS and local Parquet partitions
- Backend and frontend services restarted to pick up new data

### Part 5 — Real-Time Upload Pipeline (Kafka)

When a user uploads a photo through the web UI, it flows through the full ML pipeline in real-time:

1. **FastAPI Backend** saves the file locally and publishes `image_uploaded` event to **Kafka**
2. **Upload Consumer** picks up the event from the Kafka topic
3. Uploads the image to **HDFS** (date-partitioned: `/photos/uploads/{user}/date=YYYY-MM-DD/`)
4. Sends the image (base64-encoded) to **Ray Serve** for MobileNet V3 CNN inference
5. Generates a **sentence-transformers** embedding and updates the **HNSW** vector index
6. Writes a Parquet metadata row to both local disk and HDFS
7. Publishes `image_labeled` event back to Kafka (or `processing_failed` on error)
8. Photo immediately appears in the gallery with predicted labels and correct category

---

## Tech Stack

| Layer | Component | Technology | Purpose |
|-------|-----------|-----------|---------|
| **Storage** | HDFS | Hadoop 3.2.1 (NameNode + DataNode) | Distributed image & metadata storage |
| **Streaming** | Kafka | Apache Kafka 3.7.0 (KRaft mode, no ZooKeeper) | Event-driven real-time upload pipeline |
| **Analytics** | Spark | Apache Spark 3.5.0 (Master + Worker) | EDA, metadata processing, story generation |
| **ML Training** | Ray + PyTorch | Ray 2.9.3 + torchvision | MobileNet V3 Small fine-tuning (multi-label) |
| **ML Serving** | Ray Serve | `@serve.deployment` decorator | Real-time CNN inference endpoint |
| **Vision LM** | Ollama | LLaVA vision model | Image captioning & object detection |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 | 128-dim semantic text embeddings |
| **Vector Search** | hnswlib | HNSW algorithm | Approximate nearest neighbor search |
| **Backend** | FastAPI | Python REST API | 8 endpoints, Kafka producer, HNSW search |
| **Frontend** | React + Vite | Dark-mode dashboard | Gallery, search, stories, upload UI |
| **Infrastructure** | Docker Compose | 11 orchestrated services | Full platform orchestration |

---

## Prerequisites

- **Docker Desktop** (with at least 8 GB RAM allocated)
- **Docker Compose** v2+
- **Make** (GNU Make)
- ~5 GB free disk space (for Docker images and model artifacts)

---

## Quick Start

### 1. Clone and start all services

```bash
git clone https://github.com/pateldhruv1672/Big-photos.git
cd Big-photos

# Build and start all 11 Docker services
docker compose up -d --build
```

### 2. Initialize infrastructure

```bash
# Wait for services to be healthy, create HDFS directories and Kafka topics
make init
```

### 3. Run the full pipeline (parts 1–4 + tests)

```bash
make all
```

This runs all 5 parts sequentially:

| Command | What it does | Duration |
|---------|-------------|----------|
| `make part1` | Download MIRFLICKR images → HDFS upload → Spark EDA | ~2 min |
| `make part2` | VLM enrichment → embeddings → HNSW index | ~3 min |
| `make part3` | MobileNet V3 training (8 epochs) → Ray Serve deploy | ~5 min |
| `make part4` | Spark story generation → UI aggregates | ~1 min |
| `make test` | Smoke tests (all endpoints + artifacts) | ~30 sec |

### 4. Access the platform

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | [http://localhost:3000](http://localhost:3000) |
| **Backend API** | [http://localhost:8080/health](http://localhost:8080/health) |
| **HDFS Web UI** | [http://localhost:9870](http://localhost:9870) |
| **Ray Dashboard** | [http://localhost:8265](http://localhost:8265) |
| **Spark Master UI** | [http://localhost:8081](http://localhost:8081) |

### Run parts individually

```bash
make part1    # MIRFLICKR data ingestion + Spark EDA
make part2    # VLM enrichment + HNSW vector index
make part3    # MobileNet V3 training + Ray Serve deployment
make part4    # Spark stories + UI aggregates
make test     # Smoke tests + unit tests
```

### Tear down

```bash
docker compose down       # Stop all services (keep data)
make clean                # Stop services + delete all generated data
```

---

## Services & Ports

| # | Service | Container | Port | Description |
|---|---------|-----------|------|-------------|
| 1 | HDFS NameNode | `namenode` | 9870, 9000 | Distributed filesystem coordinator |
| 2 | HDFS DataNode | `datanode` | — | Data storage node |
| 3 | Spark Master | `spark-master` | 8081 | Distributed analytics engine |
| 4 | Spark Worker | `spark-worker` | — | Spark compute node |
| 5 | Kafka | `kafka` | 9092 | Event streaming broker (KRaft) |
| 6 | Ray Head | `ray-head` | 8265, 6379, 8000 | Ray cluster + Serve endpoint |
| 7 | Ray Worker | `ray-worker` | — | Ray compute node |
| 8 | Ollama | `ollama` | 11435 | Vision-language model server |
| 9 | Backend | `backend` | 8080 | FastAPI REST API |
| 10 | Upload Consumer | `upload-consumer` | — | Kafka consumer (ML pipeline) |
| 11 | Frontend | `frontend` | 3000 | React + Vite dashboard |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe (Kafka status, timestamp) |
| `GET` | `/gallery?limit=&category=&offset=` | Paginated photo gallery with category filter |
| `GET` | `/search?q=&k=` | HNSW semantic image search (vector similarity) |
| `GET` | `/stories?limit=` | Spark-generated story cards (grouped by location) |
| `GET` | `/metrics` | Dashboard summary (image counts, category distribution, HNSW size) |
| `GET` | `/image/{image_id}` | Serve image thumbnails (MIRFLICKR + uploads) |
| `POST` | `/upload` | Bulk image upload → Kafka event pipeline |
| `POST` | `/predict` | Proxy to Ray Serve MobileNet inference |

**Example requests:**

```bash
# Health check
curl http://localhost:8080/health

# Get gallery (first 20 nature photos)
curl "http://localhost:8080/gallery?category=nature&limit=20"

# Semantic search
curl "http://localhost:8080/search?q=sunset+over+water&k=5"

# Upload an image (triggers full Kafka → HDFS → MobileNet → HNSW pipeline)
curl -X POST http://localhost:8080/upload \
  -F "files=@my_photo.jpg" \
  -F "user_id=demo_user"
```

---

## Model Performance

**MobileNet V3 Small** fine-tuned on 80 training images (20 validation), 8 epochs:

| Metric | Score |
|--------|-------|
| F1 Micro | **0.76** |
| F1 Macro | 0.45 |
| F1 Weighted | 0.72 |

**Top performing classes (F1 score):**

| Class | F1 | Class | F1 |
|-------|-----|-------|-----|
| sky | 0.97 | plant_life | 0.95 |
| clouds | 0.94 | sunset | 0.93 |
| tree | 0.89 | water | 0.89 |
| people | 0.82 | structures | 0.80 |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNet V3 Small (pretrained ImageNet) |
| Loss function | BCEWithLogitsLoss (multi-label) |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 8 |
| Batch size | 16 |
| Input size | 224 × 224 |
| Threshold | 0.5 (sigmoid) |
| Classes | 24 (MIRFLICKR annotation categories) |
| Frozen layers | features[:8] (early convolutions) |

---

## Dataset

**MIRFLICKR-25K** — 100 real photographs selected for annotation richness.

- **24 annotation categories:** animals, baby, bird, car, clouds, dog, female, flower, food, indoor, lake, male, night, people, plant_life, portrait, river, sea, sky, structures, sunset, transport, tree, water
- **Average ~5 labels per image**
- **Image format:** 500×333 RGB JPEGs
- **Display categories** (weighted scoring): people (43), nature (25), city (21), travel (11)

---

## Real-Time Upload Pipeline

The Kafka-powered upload pipeline demonstrates a complete event-driven ML workflow:

```
User Upload ──► FastAPI ──► Kafka (image_uploaded) ──► Upload Consumer
                  │                                          │
                  │ save locally                             ├── 1. Upload to HDFS
                  │                                          ├── 2. MobileNet CNN inference (Ray Serve)
                  │                                          ├── 3. Generate embedding (sentence-transformers)
                  │                                          ├── 4. Update HNSW vector index
                  │                                          ├── 5. Write Parquet metadata
                  │                                          └── 6. Publish to Kafka (image_labeled)
                  │
                  └── Photo appears in gallery with predicted labels
```

**Kafka topics:**
- `image_uploaded` — triggered on user upload
- `image_labeled` — published after successful ML processing
- `processing_failed` — published on processing errors

---

## Project Structure

```
.
├── backend/
│   └── main.py                  # FastAPI backend (8 endpoints, Kafka producer)
├── consumer/
│   └── upload_consumer.py       # Kafka consumer (HDFS → Ray Serve → HNSW → Parquet)
├── serve/
│   └── ray_serve_app.py         # Ray Serve deployment (MobileNet V3 Small inference)
├── ray_jobs/
│   ├── train_classifier.py      # MobileNet V3 Small training (PyTorch, multi-label)
│   └── vlm_enrich.py            # VLM enrichment (Ollama + sentence-transformers)
├── spark_jobs/
│   ├── build_basic_metadata.py  # Spark: image metadata → Parquet
│   ├── build_ui_aggregates.py   # Spark: stories, gallery, search aggregates
│   └── eda.py                   # Spark: exploratory data analysis
├── scripts/
│   ├── prepare_mirflickr25k.py  # Download & prepare MIRFLICKR-25K dataset
│   ├── build_hnsw_index.py      # Build HNSW vector search index
│   ├── start_ray_serve.sh       # Ray Serve startup (with fallback)
│   ├── hdfs_init.sh             # Initialize HDFS directory structure
│   ├── create_kafka_topics.sh   # Create Kafka topics
│   └── wait_for_services.sh     # Health check waiter for all services
├── src/
│   ├── common/
│   │   ├── embeddings.py        # Embedding utility (sentence-transformers / Ollama / MD5)
│   │   └── hdfs.py              # WebHDFS REST client (no Hadoop CLI needed)
│   └── search/
│       └── hnsw_index.py        # HNSW index wrapper (hnswlib)
├── frontend/
│   └── src/
│       ├── App.jsx              # React dashboard (gallery, search, stories, upload)
│       └── App.css              # Dark-mode glassmorphism UI
├── tests/
│   ├── test_ray_serve.py        # 14 unit tests (MobileNet, embeddings, HNSW, Ray Serve)
│   └── smoke_test.py            # 12 smoke tests (all endpoints + file artifacts)
├── docker/
│   ├── backend.Dockerfile       # Python 3.10 + FastAPI + ML dependencies
│   └── frontend.Dockerfile      # Node 20 + Vite dev server
├── docker-compose.yml           # 11 services orchestration
├── Makefile                     # Pipeline orchestration (make all / part1-4 / test)
└── requirements.txt             # Python dependencies
```

---

## Testing

### Unit Tests (14 tests)

Tests for the ML pipeline components — MobileNet inference, embedding generation, HNSW index operations, and Ray Serve endpoint.

```bash
docker compose exec ray-head python tests/test_ray_serve.py
```

### Smoke Tests (12 tests)

End-to-end tests for all API endpoints, service health, and file artifact verification.

```bash
docker compose exec backend python tests/smoke_test.py
```

### Run all tests

```bash
make test
```

---

## Kafka Demo

To demonstrate the real-time Kafka pipeline during a presentation:

**Terminal 1 — Watch Kafka consumer logs:**
```bash
docker compose logs -f upload-consumer
```

**Terminal 2 — Upload a photo:**
```bash
curl -X POST http://localhost:8080/upload \
  -F "files=@any_photo.jpg" \
  -F "user_id=demo_user"
```

You will see the consumer log each step:
1. `Uploaded to HDFS: hdfs://namenode:9000/photos/uploads/demo_user/date=.../photo.jpg`
2. `Prediction: sky (0.97)` — MobileNet CNN inference result
3. `HNSW index updated for {image_id}`
4. `Metadata row saved: outputs/metadata/uploads/date=.../...parquet`
5. `Published image_labeled for {image_id}`

The uploaded photo appears instantly in the gallery at [http://localhost:3000](http://localhost:3000).

---

## HDFS Directory Layout

```
/photos/
├── raw/images/                          # Original MIRFLICKR-25K images
├── metadata/
│   ├── basic/                           # Spark-generated basic metadata
│   ├── enriched/date=YYYY-MM-DD/        # VLM-enriched metadata + embeddings
│   └── uploads/date=YYYY-MM-DD/         # Upload pipeline metadata
├── uploads/{user_id}/date=YYYY-MM-DD/   # User-uploaded images
├── models/image_classifier/             # Trained MobileNet V3 model
├── aggregates/
│   ├── final_stories/                   # Spark story cards
│   ├── gallery_metadata/                # Gallery display data
│   └── search_metadata/                 # Search index data
└── eda/                                 # EDA outputs (CSV, JSON, charts)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **WebHDFS REST API** instead of `hdfs dfs` CLI | Allows any container to access HDFS without Hadoop binaries installed |
| **BCEWithLogitsLoss** instead of CrossEntropy | Images have multiple labels simultaneously (sky + water + sunset) |
| **KRaft mode Kafka** (no ZooKeeper) | Simpler deployment, fewer services, production-ready since Kafka 3.3 |
| **sentence-transformers** for embeddings | Semantic similarity (not just keyword match) for HNSW vector search |
| **Weighted category scoring** | More specific categories (food, event) get higher weight multipliers |
| **Date-partitioned HDFS paths** | Standard big data convention, enables efficient time-range queries |
| **Ray Serve `@serve.deployment`** | Production-grade model serving with autoscaling, batching, and Ray Dashboard visibility |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Docker mount error on macOS | Restart Docker Desktop |
| Kafka not connecting | Wait 30s after `docker compose up`, then `make init` |
| Ray Serve health check failing | Run `docker compose exec -d ray-head bash scripts/start_ray_serve.sh` |
| Images not loading in gallery | Verify `make part1` completed, check `outputs/metadata/enriched/` has `.parquet` files |
| Empty category filters | Re-run `make part2` to regenerate enriched metadata with weighted categories |
| Stale data (old date partitions) | Delete old partitions: `rm -rf outputs/metadata/enriched/date=OLD_DATE` |
