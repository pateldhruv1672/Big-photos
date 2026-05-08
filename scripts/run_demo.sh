#!/usr/bin/env bash
set -e
echo "Demo endpoint checks:"
curl -sf http://localhost:8080/health
echo
curl -sf http://localhost:8080/stories || true
echo
echo "Open http://localhost:3000"
