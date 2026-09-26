"""
solution.py — High-performance rule-based traffic event detection and accident anticipation.

Conforms strictly to the WIUT Hackathon 2026 CV Track evaluation harness:
  - Part A: detect_events(video_path: str) -> list[list]
  - Part B: RiskEstimator with reset(meta) and step(frame, t_sec) -> float
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.detector import DetectorTracker
from src.rules import RuleEngine
from src.tracker import TrajectoryManager
from src.utils import merge_overlapping_events

# Official class ids (14).
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds).
RISK_HORIZON_SEC = 5.0

# Global cached detector instance to avoid repeated model loading
_GLOBAL_DETECTOR: DetectorTracker | None = None


def get_detector() -> DetectorTracker:
    """Lazy loader for the shared YOLOv8 object detector and tracker."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = DetectorTracker(model_name_or_path="yolov8n.pt")
    return _GLOBAL_DETECTOR


def detect_events(video_path: str) -> list[list]:
    """Part A — Traffic event detection.

    Args:
        video_path: Path to one .mp4 file.

    Returns:
        List of [start_sec, end_sec, label] intervals.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    duration = n_frames / float(fps) if fps > 0 else 0.0

    if n_frames <= 0 or duration <= 0:
        cap.release()
        return []

    detector = get_detector()
    detector.reset()

    traj_manager = TrajectoryManager()
    rule_engine = RuleEngine(frame_height=height, frame_width=width)

    # Frame sampling stride: ~2.5 fps gives optimal accuracy while easily staying well within time budget
    sample_stride = max(2, int(round(fps / 2.5)))

    frame_idx = 0
    while True:
        if frame_idx % sample_stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_idx / fps
            detections = detector.track_frame(frame, persist=True)
            traj_manager.update(detections, t_sec)
            active_tracks = traj_manager.get_active_tracks()
            rule_engine.process_frame(active_tracks, traj_manager, t_sec)
        else:
            # Advance video position without costly full frame decompression
            ok = cap.grab()
            if not ok:
                break

        frame_idx += 1

    cap.release()

    # Finalize any remaining events and merge overlapping segments
    raw_events = rule_engine.finalize_events(duration)
    cleaned_events = merge_overlapping_events(
        raw_events,
        merge_gap=1.5,
        min_duration=0.5,
        max_duration=duration,
    )

    return cleaned_events


class RiskEstimator:
    """Part B — Causal accident anticipation.

    Computes P(accident starts within the next 5s) strictly from past frames.
    """

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.fps: float = 25.0
        self.width: int = 1920
        self.height: int = 1080
        self.frame_idx: int = 0
        self.stride: int = 3
        self.last_score: float = 0.0

        self.detector: DetectorTracker | None = None
        self.traj_manager: TrajectoryManager = TrajectoryManager()
        self.rule_engine: RuleEngine = RuleEngine()

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        self.meta = meta
        self.fps = float(meta.get("fps", 25.0))
        self.width = int(meta.get("width", 1920))
        self.height = int(meta.get("height", 1080))
        self.frame_idx = 0
        self.last_score = 0.0

        # Sample every ~0.4s (approx 2.5 fps) in streaming mode to keep runtime safely under budget
        self.stride = max(2, int(round(self.fps / 2.5)))

        self.detector = get_detector()
        self.detector.reset()
        self.traj_manager.reset()
        self.rule_engine.reset(frame_height=self.height, frame_width=self.width)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s).

        Args:
            frame: BGR uint8 array of shape (H, W, 3).
            t_sec: timestamp of this frame in seconds.

        Returns:
            A float in [0.0, 1.0].
        """
        if self.detector is None:
            self.detector = get_detector()

        # Run detection and tracking on sampled frames
        if self.frame_idx % self.stride == 0:
            detections = self.detector.track_frame(frame, persist=True)
            self.traj_manager.update(detections, t_sec)
            active_tracks = self.traj_manager.get_active_tracks()
            self.last_score = self.rule_engine.compute_causal_risk(
                active_tracks, t_sec, horizon_sec=RISK_HORIZON_SEC
            )
        else:
            # Gentle decay between sampled frames
            self.last_score = max(0.01, self.last_score * 0.96)

        self.frame_idx += 1
        return float(self.last_score)
