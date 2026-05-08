# Distributed Photo Intelligence Platform

This build pack contains:
- 5 detailed Cursor specs
- 5 Cursor prompts
- Docker Compose template
- Makefile
- skeleton scripts/jobs/services

## Sample dataset size

By default, `scripts/generate_sample_dataset.py` and `make part1` generate **100** synthetic JPEGs (with real MIRFLICKR tags when `data/tags.zip` or `data/mirflickr/tags/` exists, otherwise the built-in tag vocabulary). To use another count: `python scripts/generate_sample_dataset.py --n <N>` (or pass `--n <N>` in the Makefile `part1` target).

## Real-image demo flow (Picsum thumbnails)

Synthetic tiles are colored placeholders with burned-in text. To show **real photo thumbnails** in the UI while keeping the same `tags.json` IDs and pipeline metadata shape:

1. `make part1` — generate `tags.json` + synthetic JPEGs + Spark basic metadata + EDA  
2. `make real-images` — download `https://picsum.photos/seed/{image_id}/256/256` into `data/mirflickr/images/{image_id}.jpg` (overwrites synthetic files only; **does not** change `tags.json`)  
3. `make part1-metadata-only` — re-run `build_basic_metadata.py` and `eda.py` so dimensions/file sizes match the new JPEGs (does **not** regenerate images)  
4. `make part2` → `make part3` → `make part4` — enrichment, training, stories/UI as usual  

Requires network access from the `ray-head` container for Picsum downloads.

## Part 2 (VLM enrichment) — first run vs later runs

The first `make part2` can be slow: Ollama runs vision enrichment over every image. Later runs are faster: if `outputs/metadata/enriched` already contains non-empty Parquet files, `ray_jobs/vlm_enrich.py` skips re-enrichment and exits successfully (the Makefile still runs the HNSW index step afterward). To force a full VLM pass again, set `FORCE_ENRICH=1` when invoking enrichment, for example:

```bash
docker compose exec -e FORCE_ENRICH=1 ray-head python ray_jobs/vlm_enrich.py
```

## Run

```bash
make all
```

## Services

- Frontend: http://localhost:3000
- Backend: http://localhost:8080/health
- HDFS UI: http://localhost:9870
- Ray Dashboard: http://localhost:8265

## Architecture Rules

- HDFS stores images and Parquet metadata.
- Kafka queues uploads.
- Spark runs EDA and UI/story aggregation.
- Ray trains and serves classifier.
- Ollama handles VLM labeling with fallback.
- HNSW via hnswlib handles vector search.
- No Postgres.
- No Qdrant.
