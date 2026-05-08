SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: env up down logs hdfs-init sample-data annotation-import shard-import metadata-canonical metadata-audit sanity-check e2e-sanity train-annotations-25k train-annotations-25k-tuned train-transfer-mobilenet-tuned compare-backbones index-existing thumbnails build-index validate-index ml-mirflickr25k part1 part2 part3 part4 ui ui-existing all test zip clean

env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi

up: env
	$(COMPOSE) up -d namenode datanode kafka spark-master spark-worker ray-head ray-worker
	$(COMPOSE) run --rm backend bash /app/scripts/wait_for_services.sh
	$(COMPOSE) run --rm backend bash /app/scripts/hdfs_init.sh

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

hdfs-init: env
	$(COMPOSE) run --rm backend bash /app/scripts/hdfs_init.sh

sample-data: env
	$(COMPOSE) run --rm backend python /app/scripts/generate_sample_dataset.py
	$(COMPOSE) run --rm backend python /app/scripts/hdfs_put_sample.py

annotation-import: env
	$(COMPOSE) run --rm backend python /app/scripts/import_annotations_metadata.py --allow-empty

shard-import: env
	$(COMPOSE) exec -T backend python /app/scripts/import_shard_metadata.py --exif-root /app/shared_data/exif_11db --tags-root /app/shared_data/tags_11gb --allow-empty

metadata-canonical: env
	$(COMPOSE) exec -T backend python /app/scripts/build_canonical_metadata.py --allow-empty

metadata-audit: env
	$(COMPOSE) exec -T backend python /app/scripts/metadata_quality_audit.py --allow-empty

sanity-check: env
	$(COMPOSE) exec -T backend python /app/scripts/sanity_check_e2e.py --api-base http://localhost:8001 --frontend-url http://host.docker.internal:5173

e2e-sanity: env
	$(COMPOSE) up -d backend frontend upload-consumer kafka ray-head
	-@curl --max-time 15 -sS -X POST http://localhost:8001/api/refresh >/dev/null || true
	python scripts/sanity_check_e2e.py --api-base http://localhost:8001 --frontend-url http://localhost:5173 --gallery-limit 20 --thumb-check-count 8

# UI path: use existing indexed data + existing local metadata, normalized into HDFS Parquet.
index-existing: env shard-import annotation-import
	$(COMPOSE) run --rm backend python /app/scripts/import_existing_metadata.py --input-dir /app/data/existing_metadata --allow-empty --scan-hdfs-if-missing false
	$(COMPOSE) run --rm backend python /app/scripts/generate_missing_thumbnails.py
	$(COMPOSE) run --rm backend python /app/scripts/build_hnsw_from_metadata.py
	$(COMPOSE) run --rm backend python /app/scripts/validate_ui_index.py

train-annotations-25k: env
	$(COMPOSE) exec -T backend python -c "from src.common import hdfs; [hdfs.delete(p, recursive=True) for p in ['/photos/metadata/mirflickr25k/imported','/photos/metadata/mirflickr25k/basic','/photos/metadata/mirflickr25k/enriched']]"
	$(COMPOSE) exec -T backend python /app/scripts/import_annotations_metadata.py --annotation-dir /app/shared_data/mirflickr25k_annotations_v080 --dataset mirflickr25k --raw-roots /photos/raw/team_gallery/images,/photos/raw/mirflickr25k/images --imported-root /photos/metadata/mirflickr25k/imported --active-root /photos/metadata/mirflickr25k/basic --include-all-label-files --allowed-labels structures,indoor,plant_life,people,sky,animals,water,food,transport,female --add-other-class --other-label other
	$(COMPOSE) exec -T backend sh -lc 'RAY_ADDRESS=local python /app/ray_jobs/vlm_enrich.py --input /photos/metadata/mirflickr25k/basic --output /photos/metadata/mirflickr25k/enriched --limit 0'
	$(COMPOSE) exec -T backend sh -lc 'RAY_ADDRESS=local python /app/ray_jobs/train_classifier.py --input /photos/metadata/mirflickr25k/enriched'

train-annotations-25k-tuned: env
	$(MAKE) train-annotations-25k
	$(COMPOSE) exec -T backend sh -lc 'RAY_ADDRESS=local python /app/ray_jobs/tune_classifier.py --input /photos/metadata/mirflickr25k/enriched --model-output /photos/models/image_classifier/model_tuned.pkl --metrics-output /photos/models/image_classifier/metrics_tuned.json --trials 20'

train-transfer-mobilenet-tuned: env
	$(COMPOSE) exec -T backend python -c "from src.common import hdfs; [hdfs.delete(p, recursive=True) for p in ['/photos/metadata/mirflickr25k/imported','/photos/metadata/mirflickr25k/basic']]"
	$(COMPOSE) exec -T backend python /app/scripts/import_annotations_metadata.py --annotation-dir /app/shared_data/mirflickr25k_annotations_v080 --dataset mirflickr25k --raw-roots /photos/raw/team_gallery/images --imported-root /photos/metadata/mirflickr25k/imported --active-root /photos/metadata/mirflickr25k/basic --include-all-label-files --allowed-labels structures,indoor,plant_life,people,sky,animals,water,food,transport,female --add-other-class --other-label other
	$(COMPOSE) exec -T backend sh -lc 'RAY_ADDRESS=local python /app/ray_jobs/tune_transfer_mobilenet.py --input /photos/metadata/mirflickr25k/basic --model-output /photos/models/image_classifier/mobilenet_tuned.json --metrics-output /photos/models/image_classifier/mobilenet_tuned_metrics.json --trials 10 --max-rows 2500'

compare-backbones: env
	$(COMPOSE) exec -T backend sh -lc 'RAY_ADDRESS=local python /app/ray_jobs/compare_backbones.py --backbones mobilenet_v3_small --trials-per-backbone 6 --max-rows 2500'

thumbnails: env
	$(COMPOSE) run --rm backend python /app/scripts/generate_missing_thumbnails.py --force

build-index: env
	$(COMPOSE) run --rm backend python /app/scripts/build_hnsw_from_metadata.py

validate-index: env
	$(COMPOSE) run --rm backend python /app/scripts/validate_ui_index.py

# Annotation-first + Ray-only enrichment/training path.
part1: env
	$(COMPOSE) run --rm spark-master spark-submit --master spark://spark-master:7077 /app/spark_jobs/build_basic_metadata.py --dataset team_gallery --input /photos/raw/team_gallery/images --output /photos/metadata/basic
	$(COMPOSE) run --rm spark-master spark-submit --master spark://spark-master:7077 /app/spark_jobs/eda.py --dataset team_gallery --input /photos/metadata/basic

part2: env
	$(COMPOSE) exec ray-head python /app/ray_jobs/vlm_enrich.py --input /photos/metadata/ui_active --output /photos/metadata/enriched

part3: env
	$(COMPOSE) exec ray-head python /app/ray_jobs/train_classifier.py --input /photos/metadata/enriched
	$(COMPOSE) exec -d ray-head bash /app/scripts/start_ray_serve.sh || true

ml-mirflickr25k: part1 part2 part3

part4 ui: env
	$(COMPOSE) up -d backend upload-consumer frontend
	$(COMPOSE) exec -T spark-master spark-submit --master spark://spark-master:7077 /app/spark_jobs/build_ui_aggregates.py || true

# Non-destructive mode: use already-indexed HDFS data without re-import/rebuild.
ui-existing: env
	@grep -q "^HDFS_NAMENODE_HTTP=http://host.docker.internal:9870" .env || \
		echo "Warning: .env is not pointing to existing host HDFS (host.docker.internal:9870)"
	$(COMPOSE) up -d backend upload-consumer frontend
	@echo "Using existing HDFS metadata/index artifacts (no re-index/re-import)."

all: up index-existing part2 part3 ui part4 test
	@echo "Big Photos is ready: frontend=http://localhost:5173 backend=http://localhost:8001"

test: env
	$(COMPOSE) run --rm backend python -m compileall /app/backend /app/consumer /app/ray_jobs /app/scripts /app/serve /app/spark_jobs /app/src /app/tests
	$(COMPOSE) run --rm backend python /app/tests/test_ray_serve.py || true
	$(COMPOSE) run --rm backend python /app/tests/test_metadata_import.py
	$(COMPOSE) run --rm backend python /app/tests/test_hnsw_index.py
	$(COMPOSE) run --rm backend python /app/tests/test_backend_health.py || true

zip:
	cd .. && zip -r big-photos-clone-updated.zip big-photos-clone-updated -x '*/__pycache__/*' '*.pyc' '*/node_modules/*' '*/.git/*'

clean:
	rm -rf outputs/eda/* state/*
