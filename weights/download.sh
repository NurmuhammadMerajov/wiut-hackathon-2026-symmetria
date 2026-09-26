#!/usr/bin/env bash
# Download model weights for WIUT Hackathon 2026 offline evaluation
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p "$DIR"

echo "Downloading YOLOv8n weights to $DIR/yolov8n.pt..."
curl -L -o "$DIR/yolov8n.pt" https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

echo "Weights downloaded successfully. Size: $(ls -lh "$DIR/yolov8n.pt" | awk '{print $5}')"
