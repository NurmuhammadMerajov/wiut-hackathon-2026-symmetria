"""
src/detector.py — High-performance YOLOv8 + ByteTrack object detection and tracking.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Relevant COCO class IDs
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
PEDESTRIAN_CLASSES = {0}
TARGET_CLASSES = list(VEHICLE_CLASSES | PEDESTRIAN_CLASSES)

CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls_id: int
    cls_name: str
    conf: float
    is_vehicle: bool
    is_pedestrian: bool


class DetectorTracker:
    """YOLOv8 nano/small object detector and ByteTrack tracker."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str | None = None,
        imgsz: int = 512,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        # Automatic device selection
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_half = self.device.startswith("cuda")

        # Resolve weights path
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        local_weight = weights_dir / model_name_or_path
        if local_weight.exists():
            resolved_model = str(local_weight)
        else:
            resolved_model = model_name_or_path

        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Please install it using 'pip install ultralytics'."
            )

        self.model = YOLO(resolved_model)
        self._temp_id_counter = 100000

    def reset(self) -> None:
        """Reset internal tracker state between videos."""
        self._temp_id_counter = 100000
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    for tracker in self.model.predictor.trackers:
                        if hasattr(tracker, "reset"):
                            tracker.reset()
        except Exception:
            pass

    def track_frame(
        self, frame: np.ndarray, persist: bool = True
    ) -> list[Detection]:
        """Run detection and tracking on a single BGR frame.

        Args:
            frame: BGR image (H, W, 3)
            persist: Keep track history across sequential frames

        Returns:
            List of active Detection objects with track_id and bounding boxes.
        """
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else None
        )

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            if cls_id not in CLASS_NAMES:
                continue

            if track_ids is not None and i < len(track_ids):
                track_id = int(track_ids[i])
            else:
                self._temp_id_counter += 1
                track_id = self._temp_id_counter

            bbox = (
                float(xyxy[i][0]),
                float(xyxy[i][1]),
                float(xyxy[i][2]),
                float(xyxy[i][3]),
            )
            conf = float(confs[i])
            cls_name = CLASS_NAMES.get(cls_id, "unknown")
            is_vehicle = cls_id in VEHICLE_CLASSES
            is_pedestrian = cls_id in PEDESTRIAN_CLASSES

            detections.append(
                Detection(
                    track_id=track_id,
                    bbox=bbox,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=conf,
                    is_vehicle=is_vehicle,
                    is_pedestrian=is_pedestrian,
                )
            )

        return detections
