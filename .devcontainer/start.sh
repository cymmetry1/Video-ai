#!/usr/bin/env bash
set -e
cd "$(git rev-parse --show-toplevel)"
mkdir -p work/uploads work/outputs
pkill -f 'uvicorn app:app' 2>/dev/null || true
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/video-ai.log 2>&1 &
echo 'Video AI starting on port 8000.'
