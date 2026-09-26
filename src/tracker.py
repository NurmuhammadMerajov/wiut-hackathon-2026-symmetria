"""
src/tracker.py — Trajectory tracking, kinematic state estimation, and traffic flow analysis.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence, Tuple

from src.detector import Detection
from src.utils import euclidean_distance, get_box_centroid


@dataclass
class TrackState:
    track_id: int
    cls_id: int
    cls_name: str
    is_vehicle: bool
    is_pedestrian: bool

    # Current frame attributes
    last_t: float = 0.0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    centroid: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/sec
    speed: float = 0.0  # magnitude in px/sec
    heading: float = 0.0  # radians

    # Stationary tracking
    first_stopped_t: float | None = None
    stationary_duration: float = 0.0
    anchor_centroid: tuple[float, float] | None = None

    # History buffers (maxlen=60 frames)
    history_t: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    history_centroid: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    history_speed: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    history_bbox: deque[tuple[float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )

    # Life cycle
    first_seen_t: float = 0.0
    missed_count: int = 0

    def update(
        self,
        bbox: tuple[float, float, float, float],
        t_sec: float,
        alpha: float = 0.5,
        drift_threshold: float = 25.0,
        stop_speed_threshold: float = 12.0,
    ) -> None:
        """Update track with new bounding box at timestamp t_sec."""
        new_centroid = get_box_centroid(bbox)
        dt = t_sec - self.last_t if self.last_t > 0 else 0.0

        if dt > 1e-4:
            raw_vx = (new_centroid[0] - self.centroid[0]) / dt
            raw_vy = (new_centroid[1] - self.centroid[1]) / dt

            # Exponential Moving Average for velocity to suppress detection jitter
            vx = alpha * raw_vx + (1.0 - alpha) * self.velocity[0]
            vy = alpha * raw_vy + (1.0 - alpha) * self.velocity[1]
            self.velocity = (vx, vy)
            self.speed = math.hypot(vx, vy)
            if self.speed > 5.0:
                self.heading = math.atan2(vy, vx)
        elif self.last_t == 0.0:
            self.first_seen_t = t_sec

        self.last_t = t_sec
        self.last_bbox = bbox
        self.centroid = new_centroid
        self.missed_count = 0

        # Update history
        self.history_t.append(t_sec)
        self.history_centroid.append(new_centroid)
        self.history_speed.append(self.speed)
        self.history_bbox.append(bbox)

        # Stationary logic
        if self.speed < stop_speed_threshold:
            if self.anchor_centroid is None:
                self.anchor_centroid = new_centroid
                self.first_stopped_t = t_sec
                self.stationary_duration = 0.0
            else:
                dist_to_anchor = euclidean_distance(new_centroid, self.anchor_centroid)
                if dist_to_anchor < drift_threshold:
                    if self.first_stopped_t is not None:
                        self.stationary_duration = t_sec - self.first_stopped_t
                else:
                    # Drifted beyond threshold: re-anchor
                    self.anchor_centroid = new_centroid
                    self.first_stopped_t = t_sec
                    self.stationary_duration = 0.0
        else:
            self.first_stopped_t = None
            self.anchor_centroid = None
            self.stationary_duration = 0.0


class TrajectoryManager:
    """Manages active tracks, traffic flow direction field, and scene road mask."""

    def __init__(
        self,
        max_missed_frames: int = 15,
        grid_size: int = 64,
    ) -> None:
        self.tracks: dict[int, TrackState] = {}
        self.max_missed_frames = max_missed_frames
        self.grid_size = grid_size

        # Spatial grid of flow directions: grid_cell -> list of (vx, vy)
        self.flow_grid: dict[tuple[int, int], list[tuple[float, float]]] = {}

        # Estimated road bounds from vehicle traffic envelope
        self.road_y_min = float("inf")
        self.road_y_max = 0.0

    def reset(self) -> None:
        """Clear all tracks and state."""
        self.tracks.clear()
        self.flow_grid.clear()
        self.road_y_min = float("inf")
        self.road_y_max = 0.0

    def update(self, detections: list[Detection], t_sec: float) -> None:
        """Update trajectory states with new detections at timestamp t_sec."""
        detected_ids = set()

        for det in detections:
            tid = det.track_id
            detected_ids.add(tid)

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(
                    track_id=tid,
                    cls_id=det.cls_id,
                    cls_name=det.cls_name,
                    is_vehicle=det.is_vehicle,
                    is_pedestrian=det.is_pedestrian,
                )

            track = self.tracks[tid]
            track.update(det.bbox, t_sec)

            # Update road vertical extent based on vehicle ground contacts
            if det.is_vehicle:
                y2 = det.bbox[3]
                self.road_y_min = min(self.road_y_min, y2 - 40)
                self.road_y_max = max(self.road_y_max, y2)

                # If vehicle is moving reliably, update local traffic flow grid
                if track.speed > 25.0:
                    gx = int(track.centroid[0] // self.grid_size)
                    gy = int(track.centroid[1] // self.grid_size)
                    cell = (gx, gy)
                    if cell not in self.flow_grid:
                        self.flow_grid[cell] = []
                    # Keep latest flow samples
                    self.flow_grid[cell].append(track.velocity)
                    if len(self.flow_grid[cell]) > 40:
                        self.flow_grid[cell].pop(0)

        # Mark missed tracks and prune stale ones
        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in detected_ids:
                track.missed_count += 1
                if track.missed_count > self.max_missed_frames:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

    def get_active_tracks(self) -> list[TrackState]:
        """Return tracks that were updated recently."""
        return [t for t in self.tracks.values() if t.missed_count <= 2]

    def get_expected_flow_direction(
        self, centroid: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Estimate the expected traffic flow vector (vx, vy) around a point."""
        gx = int(centroid[0] // self.grid_size)
        gy = int(centroid[1] // self.grid_size)

        # Look in current cell and adjacent 8 neighbors
        sample_vectors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (gx + dx, gy + dy)
                if cell in self.flow_grid:
                    sample_vectors.extend(self.flow_grid[cell])

        if not sample_vectors or len(sample_vectors) < 3:
            return None

        avg_vx = sum(v[0] for v in sample_vectors) / len(sample_vectors)
        avg_vy = sum(v[1] for v in sample_vectors) / len(sample_vectors)
        norm = math.hypot(avg_vx, avg_vy)
        if norm < 5.0:
            return None
        return (avg_vx / norm, avg_vy / norm)

    def is_on_roadway(self, point: tuple[float, float], frame_height: int) -> bool:
        """Check if a point is within the carriageway."""
        py = point[1]
        if self.road_y_min < float("inf") and self.road_y_max > 0:
            return (self.road_y_min <= py <= self.road_y_max)
        # Fallback default: roadway typically occupies 30% to 95% of vertical frame
        return (0.30 * frame_height <= py <= 0.95 * frame_height)
