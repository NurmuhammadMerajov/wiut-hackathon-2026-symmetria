"""
src/utils.py — Geometric calculations, event merging, and helper routines.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple


def compute_aabb_intersection(
    box_a: Sequence[float], box_b: Sequence[float]
) -> float:
    """Compute the intersection area between two Axis-Aligned Bounding Boxes.

    Boxes are [x1, y1, x2, y2].
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def compute_box_area(box: Sequence[float]) -> float:
    """Compute area of [x1, y1, x2, y2]."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return float(w * h)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    inter = compute_aabb_intersection(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_a = compute_box_area(box_a)
    area_b = compute_box_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def compute_ios(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Smaller bounding box area (IoS).

    Useful when detecting collisions between objects of disparate sizes
    (e.g., truck and bicycle or car and pedestrian).
    """
    inter = compute_aabb_intersection(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_min = min(compute_box_area(box_a), compute_box_area(box_b))
    return inter / area_min if area_min > 0.0 else 0.0


def get_box_centroid(box: Sequence[float]) -> Tuple[float, float]:
    """Return the center point (cx, cy) of [x1, y1, x2, y2]."""
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def get_box_bottom_center(box: Sequence[float]) -> Tuple[float, float]:
    """Return the bottom-center point (cx, y2) representing ground contact."""
    return ((box[0] + box[2]) * 0.5, float(box[3]))


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def cosine_similarity(v1: Sequence[float], v2: Sequence[float], eps: float = 1e-6) -> float:
    """Compute cosine similarity between two 2D vectors: (v1 . v2) / (|v1| * |v2|)."""
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 < eps or norm2 < eps:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return dot / (norm1 * norm2)


def merge_overlapping_events(
    events: list[list],
    merge_gap: float = 1.0,
    min_duration: float = 0.5,
    max_duration: float | None = None,
) -> list[list]:
    """Clean, filter, and merge same-class event segments.

    Enforces the competition rules:
    - Segments of the SAME class must not overlap.
    - If two segments of the same class are separated by <= merge_gap seconds,
      they are merged into a single continuous segment.
    - Segments with duration < min_duration seconds are filtered out as noise.
    - Start and end seconds are bounded by [0, max_duration].
    - Returns a list of [start_sec, end_sec, label] sorted by start_sec.
    """
    if not events:
        return []

    # Filter invalid intervals and group by class
    by_class: dict[str, list[list[float]]] = {}
    for ev in events:
        if len(ev) != 3:
            continue
        try:
            s, e, label = float(ev[0]), float(ev[1]), str(ev[2])
        except (ValueError, TypeError):
            continue

        if max_duration is not None:
            e = min(e, max_duration)
        if s < 0.0:
            s = 0.0

        if s >= e:
            continue

        by_class.setdefault(label, []).append([s, e])

    merged_all: list[list] = []

    for label, intervals in by_class.items():
        # Sort by start time, then end time
        intervals.sort(key=lambda x: (x[0], x[1]))

        merged_class: list[list[float]] = []
        for interval in intervals:
            if not merged_class:
                merged_class.append(interval)
                continue

            last = merged_class[-1]
            # Check overlap or small gap
            if interval[0] <= last[1] + merge_gap:
                last[1] = max(last[1], interval[1])
            else:
                merged_class.append(interval)

        # Apply min_duration filter
        for s, e in merged_class:
            if (e - s) >= min_duration:
                merged_all.append([round(s, 3), round(e, 3), label])

    # Sort globally by start time
    merged_all.sort(key=lambda x: (x[0], x[1], x[2]))
    return merged_all
