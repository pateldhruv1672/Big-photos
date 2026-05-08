#!/usr/bin/env bash
set -e
docker compose exec -T ollama ollama pull llava || docker compose exec -T ollama ollama pull moondream
