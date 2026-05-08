.PHONY: up down logs init part1 part1-metadata-only real-images part2 part3 part4 part5 all demo test clean

# ─── Infrastructure ───────────────────────────────────────────────────────────

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	rm -rf outputs/* models/* data/mirflickr/
	docker compose down -v

logs:
	docker compose logs -f --tail=100

# ─── Init (wait + HDFS dirs + Kafka topics) ───────────────────────────────────

init:
	bash scripts/wait_for_services.sh
	bash scripts/hdfs_init.sh
	bash scripts/create_kafka_topics.sh

# ─── Part 1: HDFS + Spark EDA ────────────────────────────────────────────────
# Place the MIRFLICKR tags zip at data/tags.zip before running for real tags.
# The script reads tags/{id%10}/{id}.txt from the zip automatically.

part1: up init
	@echo "=== Part 1: Preparing MIRFLICKR-25K dataset (100 real images + annotations) ==="
	docker compose exec ray-head python scripts/prepare_mirflickr25k.py --n 100
	@echo "=== Part 1: Building basic metadata ==="
	docker compose exec spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		spark_jobs/build_basic_metadata.py
	@echo "=== Part 1: Running EDA ==="
	docker compose exec spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		spark_jobs/eda.py
	@echo "Part 1 complete. EDA outputs in outputs/eda/"

# ─── Part 1 (metadata + EDA only): refresh Spark outputs without regenerating images ──

part1-metadata-only: up init
	@echo "=== Part 1 (metadata + EDA only — synthetic/real JPEGs unchanged) ==="
	docker compose exec spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		spark_jobs/build_basic_metadata.py
	@echo "=== Part 1: Running EDA ==="
	docker compose exec spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		spark_jobs/eda.py
	@echo "Metadata + EDA refreshed. Outputs in outputs/eda/"

# ─── Replace synthetic tiles with Picsum thumbnails (same IDs as tags.json) ────────

real-images:
	docker compose exec ray-head python scripts/download_real_demo_images.py

# ─── Part 2: VLM Enrichment ──────────────────────────────────────────────────

# part2: vlm_enrich.py skips when outputs/metadata/enriched has Parquet (unless FORCE_ENRICH=1).
part2: up init
	@echo "=== Part 2: Pulling Ollama model (best-effort) ==="
	bash scripts/pull_ollama_model.sh || true
	@echo "=== Part 2: Running VLM enrichment ==="
	docker compose exec ray-head python ray_jobs/vlm_enrich.py
	@echo "=== Part 2: Building HNSW index ==="
	docker compose exec ray-head python scripts/build_hnsw_index.py
	@echo "Part 2 complete."

# ─── Part 3: Ray Training + Ray Serve ────────────────────────────────────────

part3: up init
	@echo "=== Part 3: Installing PyTorch (MobileNet V3 Small) ==="
	docker compose exec ray-head pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "=== Part 3: Training MobileNet V3 Small classifier ==="
	docker compose exec ray-head python ray_jobs/train_classifier.py
	@echo "=== Part 3: Starting Ray Serve ==="
	docker compose exec -d ray-head bash scripts/start_ray_serve.sh
	@echo "Waiting for Ray Serve to start..."
	@sleep 10
	@echo "Part 3 complete. Ray Serve at http://localhost:8000"

# ─── Part 4: Web App + Spark Stories ─────────────────────────────────────────

part4: up init
	@echo "=== Part 4: Building UI aggregates and stories ==="
	docker compose exec spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		spark_jobs/build_ui_aggregates.py
	@echo "=== Part 4: Restarting backend and consumer ==="
	docker compose restart backend upload-consumer frontend
	@echo "Waiting for services to restart..."
	@sleep 5
	@echo "Part 4 complete. Frontend at http://localhost:3000"

# ─── Part 5: Unit + integration tests ────────────────────────────────────────

part5:
	@echo "=== Part 5: Running tests ==="
	docker compose exec ray-head python tests/test_ray_serve.py
	@echo "Part 5 complete."

# ─── Full pipeline ────────────────────────────────────────────────────────────

all: part1 part2 part3 part4 test
	@echo ""
	@echo "╔══════════════════════════════════════════╗"
	@echo "║  Distributed Photo Intelligence — READY  ║"
	@echo "║  Frontend  : http://localhost:3000        ║"
	@echo "║  Backend   : http://localhost:8080/health ║"
	@echo "║  Ray Serve : http://localhost:8000        ║"
	@echo "║  HDFS UI   : http://localhost:9870        ║"
	@echo "║  Ray UI    : http://localhost:8265        ║"
	@echo "╚══════════════════════════════════════════╝"

# ─── Smoke tests ─────────────────────────────────────────────────────────────

test:
	@echo "=== Running smoke tests ==="
	docker compose exec backend python tests/smoke_test.py || python tests/smoke_test.py

# ─── Demo ────────────────────────────────────────────────────────────────────

demo:
	bash scripts/run_demo.sh
