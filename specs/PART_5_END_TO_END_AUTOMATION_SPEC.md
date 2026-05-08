# Part 5: End-to-End Automation Spec


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
Create one command that builds, launches, initializes, runs, tests, and demos the full project.

## Required Commands
```bash
make all
make demo
make test
make down
```

## `make all` Must Do
1. Build Docker images.
2. Start Docker Compose.
3. Wait for HDFS, Spark, Kafka, Ray, Ollama, backend.
4. Generate sample dataset if MIRFLICKR is missing.
5. Initialize HDFS directories.
6. Run Spark metadata job.
7. Run Spark EDA job.
8. Run VLM enrichment job.
9. Train Ray classifier.
10. Start Ray Serve.
11. Start backend, consumer, frontend.
12. Build HNSW index.
13. Run Spark story aggregation.
14. Run smoke tests.

## Success Criteria
- No manual input.
- UI available at `http://localhost:3000`.
- Backend health at `http://localhost:8080/health`.
- Ray Serve available at `http://localhost:8000`.
- EDA outputs and stories exist.
