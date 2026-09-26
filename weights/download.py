"""
weights/download.py — Cross-platform Python helper to download model weights.
"""
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = WEIGHTS_DIR / "yolov8n.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_FILE.exists() and WEIGHTS_FILE.stat().st_size > 1000000:
        print(f"Weights already exist at {WEIGHTS_FILE}")
        return

    print(f"Downloading YOLOv8n weights from {URL} to {WEIGHTS_FILE}...")
    urllib.request.urlretrieve(URL, str(WEIGHTS_FILE))
    size_mb = WEIGHTS_FILE.stat().st_size / (1024 * 1024)
    print(f"Successfully downloaded weights: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
