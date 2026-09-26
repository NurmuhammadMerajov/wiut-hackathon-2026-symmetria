"""
src/rules.py — Algorithmic rule engines for traffic events and causal TTC risk estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from src.tracker import TrackState, TrajectoryManager
from src.utils import (
    compute_aabb_intersection,
    compute_box_area,
    compute_ios,
    compute_iou,
    cosine_similarity,
    euclidean_distance,
    get_box_bottom_center,
    get_box_centroid,
)


@dataclass
class OngoingEvent:
    label: str
    start_t: float
    last_active_t: float
    track_ids: set[int]
    confirmed: bool = False
    metadata: dict = field(default_factory=dict)


class RuleEngine:
    """Evaluates rule-based traffic events and causal Time-To-Collision (TTC)."""

    def __init__(self, frame_height: int = 1080, frame_width: int = 1920) -> None:
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Active ongoing events: event_key -> OngoingEvent
        self.active_events: dict[str, OngoingEvent] = {}
        # Finalized events: list of [start_sec, end_sec, label]
        self.completed_events: list[list] = []

        # Tracking wrong-way durations: track_id -> [start_t, last_t]
        self.wrong_way_tracker: dict[int, list[float]] = {}

        # Tracking jaywalking durations: track_id -> [start_t, last_t]
        self.jaywalking_tracker: dict[int, list[float]] = {}

        # Tracking stopped vehicles that reached 10s: track_id -> start_t
        self.stopped_vehicle_tracker: dict[int, float] = {}

        # Causal risk state
        self.prev_risk_score: float = 0.0
        self.last_risk_t: float = 0.0

    def reset(self, frame_height: int = 1080, frame_width: int = 1920) -> None:
        """Reset state between videos."""
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.active_events.clear()
        self.completed_events.clear()
        self.wrong_way_tracker.clear()
        self.jaywalking_tracker.clear()
        self.stopped_vehicle_tracker.clear()
        self.prev_risk_score = 0.0
        self.last_risk_t = 0.0

    def process_frame(
        self,
        tracks: list[TrackState],
        trajectory_manager: TrajectoryManager,
        t_sec: float,
    ) -> None:
        """Evaluate Part A event rules on active tracks at current timestamp t_sec."""
        active_ids = {t.track_id for t in tracks}
        tracks_by_id = {t.track_id: t for t in tracks}

        # -------------------------------------------------------------
        # 1. STOPPED VEHICLE (stationary on carriageway >= 10s)
        # -------------------------------------------------------------
        for tr in tracks:
            if not tr.is_vehicle:
                continue

            tid = tr.track_id
            if tr.stationary_duration >= 10.0:
                # Confirmed stopped vehicle
                if tid not in self.stopped_vehicle_tracker:
                    start_t = tr.first_stopped_t if tr.first_stopped_t is not None else (t_sec - 10.0)
                    self.stopped_vehicle_tracker[tid] = start_t
            else:
                # If was stopped >= 10s and now resumed motion or displaced
                if tid in self.stopped_vehicle_tracker:
                    start_t = self.stopped_vehicle_tracker.pop(tid)
                    end_t = t_sec
                    if end_t - start_t >= 10.0:
                        self.completed_events.append([round(start_t, 3), round(end_t, 3), "stopped_vehicle"])

        # -------------------------------------------------------------
        # 2. WRONG WAY (velocity opposite to dominant traffic flow)
        # -------------------------------------------------------------
        for tr in tracks:
            if not tr.is_vehicle or tr.speed < 20.0:
                continue

            tid = tr.track_id
            flow = trajectory_manager.get_expected_flow_direction(tr.centroid)
            if flow is not None:
                cos_sim = cosine_similarity(tr.velocity, flow)
                # Angle > 120 degrees against flow
                if cos_sim < -0.5:
                    if tid not in self.wrong_way_tracker:
                        self.wrong_way_tracker[tid] = [t_sec, t_sec]
                    else:
                        self.wrong_way_tracker[tid][1] = t_sec
                else:
                    # Vehicle corrected direction
                    if tid in self.wrong_way_tracker:
                        st, et = self.wrong_way_tracker.pop(tid)
                        if et - st >= 1.5:
                            self.completed_events.append([round(st, 3), round(et, 3), "wrong_way"])

        # Clean up stale wrong-way tracks that left the frame
        stale_ww = [tid for tid in self.wrong_way_tracker if tid not in active_ids]
        for tid in stale_ww:
            st, et = self.wrong_way_tracker.pop(tid)
            if et - st >= 1.5:
                self.completed_events.append([round(st, 3), round(et, 3), "wrong_way"])

        # -------------------------------------------------------------
        # 3. JAYWALKING (pedestrian on carriageway outside crossing)
        # -------------------------------------------------------------
        for tr in tracks:
            if not tr.is_pedestrian:
                continue

            tid = tr.track_id
            feet_pt = get_box_bottom_center(tr.last_bbox)
            if trajectory_manager.is_on_roadway(feet_pt, self.frame_height):
                if tid not in self.jaywalking_tracker:
                    self.jaywalking_tracker[tid] = [t_sec, t_sec]
                else:
                    self.jaywalking_tracker[tid][1] = t_sec
            else:
                if tid in self.jaywalking_tracker:
                    st, et = self.jaywalking_tracker.pop(tid)
                    if et - st >= 1.0:
                        self.completed_events.append([round(st, 3), round(et, 3), "jaywalking"])

        # Clean up stale jaywalking tracks that left the frame
        stale_jay = [tid for tid in self.jaywalking_tracker if tid not in active_ids]
        for tid in stale_jay:
            st, et = self.jaywalking_tracker.pop(tid)
            if et - st >= 1.0:
                self.completed_events.append([round(st, 3), round(et, 3), "jaywalking"])

        # -------------------------------------------------------------
        # 4. ACCIDENT (AABB intersection + abrupt speed/centroid drop)
        # -------------------------------------------------------------
        n_tracks = len(tracks)
        for i in range(n_tracks):
            tr1 = tracks[i]
            for j in range(i + 1, n_tracks):
                tr2 = tracks[j]

                # At least one must be a vehicle
                if not (tr1.is_vehicle or tr2.is_vehicle):
                    continue

                event_key = f"acc_{min(tr1.track_id, tr2.track_id)}_{max(tr1.track_id, tr2.track_id)}"
                iou = compute_iou(tr1.last_bbox, tr2.last_bbox)
                ios = compute_ios(tr1.last_bbox, tr2.last_bbox)

                # Collision condition: bounding boxes overlap
                is_overlapping = (iou > 0.05) or (ios > 0.15)

                if is_overlapping:
                    # Check pre-collision kinetic energy / speed
                    speed1_prior = (
                        max(list(tr1.history_speed)[:-2])
                        if len(tr1.history_speed) > 3
                        else tr1.speed
                    )
                    speed2_prior = (
                        max(list(tr2.history_speed)[:-2])
                        if len(tr2.history_speed) > 3
                        else tr2.speed
                    )
                    max_prior_speed = max(speed1_prior, speed2_prior)

                    # Abrupt speed drop: current speed is significantly lower than prior
                    speed_drop = (
                        (tr1.speed < 0.45 * speed1_prior if speed1_prior > 15.0 else True)
                        and (tr2.speed < 0.45 * speed2_prior if speed2_prior > 15.0 else True)
                    )

                    if max_prior_speed > 15.0 and speed_drop:
                        if event_key not in self.active_events:
                            # Start accident event
                            impact_t = t_sec
                            if len(tr1.history_t) > 2:
                                impact_t = tr1.history_t[-2]
                            self.active_events[event_key] = OngoingEvent(
                                label="accident",
                                start_t=impact_t,
                                last_active_t=t_sec,
                                track_ids={tr1.track_id, tr2.track_id},
                                confirmed=True,
                            )
                        else:
                            self.active_events[event_key].last_active_t = t_sec

        # Check and finalize accident events where objects cleared or stopped
        stale_events = []
        for key, ev in self.active_events.items():
            if ev.label == "accident":
                # If neither track is still active or time elapsed
                if (t_sec - ev.last_active_t) > 3.0:
                    stale_events.append(key)
                    end_t = ev.last_active_t + 2.0
                    if end_t - ev.start_t >= 2.0:
                        self.completed_events.append([round(ev.start_t, 3), round(end_t, 3), "accident"])

        for key in stale_events:
            del self.active_events[key]

    def finalize_events(self, video_duration: float) -> list[list]:
        """Finalize all remaining active events at end of video."""
        # Flush remaining stopped vehicles
        for tid, start_t in self.stopped_vehicle_tracker.items():
            if video_duration - start_t >= 10.0:
                self.completed_events.append([round(start_t, 3), round(video_duration, 3), "stopped_vehicle"])
        self.stopped_vehicle_tracker.clear()

        # Flush remaining wrong-way vehicles
        for tid, (st, et) in self.wrong_way_tracker.items():
            if et - st >= 1.5:
                self.completed_events.append([round(st, 3), round(et, 3), "wrong_way"])
        self.wrong_way_tracker.clear()

        # Flush remaining jaywalking
        for tid, (st, et) in self.jaywalking_tracker.items():
            if et - st >= 1.0:
                self.completed_events.append([round(st, 3), round(et, 3), "jaywalking"])
        self.jaywalking_tracker.clear()

        # Flush remaining active accidents
        for key, ev in self.active_events.items():
            if ev.confirmed:
                end_t = min(video_duration, max(ev.last_active_t + 2.0, ev.start_t + 3.0))
                self.completed_events.append([round(ev.start_t, 3), round(end_t, 3), ev.label])
        self.active_events.clear()

        return self.completed_events

    def compute_causal_risk(
        self,
        tracks: list[TrackState],
        t_sec: float,
        horizon_sec: float = 5.0,
    ) -> float:
        """Part B: Causal Time-To-Collision (TTC) risk estimation.

        Strictly causal: uses only tracks and positions up to t_sec.
        Returns a risk score in [0.0, 1.0].
        If min TTC <= 5.0s along intersecting trajectories, returns > 0.70.
        """
        min_ttc = float("inf")
        imminent_threat = False
        n_tracks = len(tracks)

        for i in range(n_tracks):
            tr1 = tracks[i]
            for j in range(i + 1, n_tracks):
                tr2 = tracks[j]

                # Focus on vehicle-vehicle and vehicle-pedestrian pairs
                if not (tr1.is_vehicle or tr2.is_vehicle):
                    continue

                # Relative position and distance
                p1 = tr1.centroid
                p2 = tr2.centroid
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = math.hypot(dx, dy)

                if dist < 1.0:
                    continue

                # Relative velocity: v2 - v1
                dvx = tr2.velocity[0] - tr1.velocity[0]
                dvy = tr2.velocity[1] - tr1.velocity[1]
                rel_speed = math.hypot(dvx, dvy)

                if rel_speed < 8.0:
                    continue

                # Closing speed: - (r . v_rel) / |r|
                closing_speed = -(dx * dvx + dy * dvy) / dist

                # Must be closing in on each other
                if closing_speed > 10.0:
                    # Time to collision range estimate
                    ttc = dist / closing_speed

                    # Compute minimum projected distance of closest approach
                    # t_closest = - (r . v_rel) / |v_rel|^2
                    t_closest = -(dx * dvx + dy * dvy) / (rel_speed * rel_speed)
                    if t_closest > 0:
                        closest_dx = dx + dvx * t_closest
                        closest_dy = dy + dvy * t_closest
                        d_closest = math.hypot(closest_dx, closest_dy)

                        # Bounding box extent
                        w1 = tr1.last_bbox[2] - tr1.last_bbox[0]
                        h1 = tr1.last_bbox[3] - tr1.last_bbox[1]
                        w2 = tr2.last_bbox[2] - tr2.last_bbox[0]
                        h2 = tr2.last_bbox[3] - tr2.last_bbox[1]
                        combined_radius = 0.5 * (max(w1, h1) + max(w2, h2))

                        # Check if trajectory projected closest approach is a collision
                        if d_closest < 1.35 * combined_radius:
                            if ttc < min_ttc:
                                min_ttc = ttc
                                if ttc <= horizon_sec:
                                    imminent_threat = True

        # Calculate instantaneous risk
        if imminent_threat and min_ttc <= horizon_sec:
            # Requirements: return score > 0.7 if TTC <= 5.0 seconds
            # Scale from 0.72 up to 0.98 as TTC decreases to 0
            fraction = max(0.0, min(1.0, (horizon_sec - min_ttc) / horizon_sec))
            instant_risk = 0.72 + 0.26 * fraction
        elif min_ttc <= horizon_sec * 1.6:
            # Scaled down for TTC in (5.0s, 8.0s]
            fraction = max(0.0, min(1.0, (horizon_sec * 1.6 - min_ttc) / (horizon_sec * 0.6)))
            instant_risk = 0.20 + 0.45 * fraction
        else:
            # Baseline low risk
            instant_risk = 0.02

        # Temporal smoothing with continuous-time exponential decay
        dt = t_sec - self.last_risk_t if self.last_risk_t > 0 else 0.0
        self.last_risk_t = t_sec

        if dt > 0.0:
            decay = math.exp(-dt / 0.8)
        else:
            decay = 0.88

        smoothed_risk = max(instant_risk, self.prev_risk_score * decay)
        smoothed_risk = max(0.0, min(1.0, smoothed_risk))
        self.prev_risk_score = smoothed_risk

        return float(smoothed_risk)
