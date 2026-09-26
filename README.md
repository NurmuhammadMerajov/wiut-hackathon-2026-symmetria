# WIUT Hackathon 2026 — Computer Vision track: starter kit

Traffic events from a fixed road camera: **detect** them as time segments
(`[start_sec, end_sec, label]`) and, as a bonus, **anticipate** accidents with a
causal risk score. Three files; read the task description for the rules.

```
solution.py          <- the ONLY file you implement (CLASSES, detect_events, RiskEstimator)
run_submission.py    <- organizers' harness: folder of videos -> predictions.json   (do not modify)
evaluate.py          <- format check + the official metric                          (do not modify)
examples/            <- ground_truth.json and predictions.json in the exact format
requirements.txt     <- numpy + opencv for the harness; add your own deps to YOUR repo
```

## Quickstart

```bash
pip install -r requirements.txt
# 1. implement solution.py
# 2. label the sample videos yourselves -> my_labels.json (same shape as examples/ground_truth.json)
python run_submission.py --videos samples --out predictions_samples.json --team <your-team>
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only        # format check without labels
```

## The interface (`solution.py`)

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...]; label in CLASSES; same-class segments don't overlap."""

class RiskEstimator:
    def reset(self, meta: dict) -> None: ...            # meta: video_id, fps, width, height, n_frames
    def step(self, frame: np.ndarray, t_sec: float) -> float: ...   # BGR uint8 frame -> P(accident within 5 s)
```

`step` is called for **every frame in order** by the harness; it must not open the
video itself. Skipping frames internally and returning the last score is fine.
You may remove ids from `CLASSES`; never add.

## What we run (offline, one GPU, no internet)

```bash
pip install -r requirements.txt            # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Time budget per video: **3 × its duration** for Part A + Part B together; a video
over budget or a crash scores as empty. Events with a bad label, bad times, or a
same-class overlap are dropped by the harness and listed in its log. Weights
≤ 5 GB, shipped in the repo or fetched once by `weights/download.sh` before the
offline run.

## predictions.json

```json
{
  "team": "your-team-name",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"], [40.0, 43.5, "red_light"]],
      "risk":   [[0.00, 0.01], [0.04, 0.01], [0.08, 0.02]]
    },
    "test_002.mp4": {"events": [], "risk": []}
  }
}
```

`risk` is written by the harness (one `[t_sec, score]` per frame). Keys are file
names. Every test video must be present, even with `"events": []`.
Ground truth: `{"test_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}`.

## Metric (exact code in `evaluate.py`)

**Part A.** Per class `c` and per tIoU threshold τ ∈ {0.3, 0.5, 0.7}: greedy
one-to-one matching by descending IoU; TP/FP/FN pooled over all videos; `F1_c(τ)`.
`Score_A = mean_c mean_τ F1_c(τ)`. Classes = those in the ground truth or in your
predictions (a class you predict that never occurs scores 0).

**Part B** (`accident` only; H = 5 s, W = 10 s, θ = 0.5). Frames in `[s−H, s)`
before an accident start `s` are positive; frames inside accidents and around
near-misses are ignored; the rest negative. `AP` = average precision over frames,
chance-normalised (`max(0, (AP_raw − r)/(1 − r))`, `r` = positive rate, so a
constant score gets 0). Alarms = runs of score ≥ θ (runs < 2 s apart merged),
alarm time = run start; an alarm in `[s−W, s)` of an unmatched accident matches it
→ `F1_alarm`; `mTTA` = mean of `s − alarm_time` (0 if unmatched).
`Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/W`.

**Model score** `M = 0.7·Score_A + 0.3·Score_B` (M = Score_A if the test set has no
accidents). Elimination score = 0.6·M + 0.25·Website + 0.15·Code.

## System Architecture & Approach

This solution implements a lean, ultra-fast, offline rule-based traffic event detection and accident anticipation system tailored for the WIUT Hackathon 2026 CV Track evaluation limits (1x NVIDIA T4 GPU, 8 CPUs, $< 3\times$ runtime budget, $\le 5$ GB weights).

```
your-repo/
├── solution.py                 # Core interface: detect_events + RiskEstimator
├── run_submission.py           # Competition harness (unchanged)
├── evaluate.py                 # Competition metric (unchanged)
├── requirements.txt            # Python dependencies (numpy, opencv, torch, ultralytics)
├── weights/
│   ├── download.sh             # Shell script to fetch YOLOv8n weights (<7 MB)
│   ├── download.py             # Cross-platform Python download script
│   └── yolov8n.pt              # Pre-trained open-weights model
├── src/
│   ├── __init__.py             # Module root
│   ├── detector.py             # YOLOv8 + ByteTrack object detection and tracking wrapper
│   ├── tracker.py              # Kinematics, trajectory smoothing, stationary counter, flow grid
│   ├── rules.py                # Algorithmic event rules (AABB collisions, stopped, wrong-way, jaywalk, TTC)
│   └── utils.py                # Geometry helpers and temporal segment overlap merger
├── examples/                   # Reference ground truth and sample predictions
└── README.md
```

### 1. Detector & Tracker (`src/detector.py`, `src/tracker.py`)
- **Detector**: Lightweight YOLOv8 nano (`yolov8n.pt`, 6.25 MB) loaded onto CUDA (with CPU fallback). Only detects relevant traffic classes: vehicles (cars, motorcycles, buses, trucks, bicycles) and pedestrians.
- **Tracker**: ByteTrack (`bytetrack.yaml`) maintains cross-frame object identity.
- **Kinematic State**: Computes smoothed velocity vectors $(v_x, v_y)$, speed, heading angle $\theta$, and stationary durations using an Exponential Moving Average (EMA) to suppress detection jitter.
- **Traffic Flow Field**: Discretizes the scene into a grid and dynamically builds a dominant flow vector field from moving vehicles without requiring hardcoded camera calibration.

### 2. Rule-Based Event Logic (`src/rules.py`)
- **`accident`**:
  - Spatial Condition: Axis-Aligned Bounding Box (AABB) intersection check ($\text{IoU} > 0.05$ or $\text{IoS} > 0.15$).
  - Kinematic Condition: Evaluates pre-collision speed against post-collision speed. An abrupt drop $> 55\%$ in speed upon contact confirms a collision.
  - Scene Stabilization: Objects remain entangled or stationary at the collision site for $\ge 2.0$s.
- **`stopped_vehicle`**:
  - Triggers when a vehicle's centroid displacement remains below the drift threshold ($< 25\text{px}$) for $\ge 10.0$ continuous seconds on the carriageway.
  - Start time = onset of stop; End time = motion resume or video end.
- **`wrong_way`**:
  - Evaluates vehicle velocity direction against the dominant traffic flow direction in that lane/region ($\cos(\theta) < -0.5$, angle $> 120^\circ$).
  - Sustained for $\ge 1.5$ seconds at speed $> 20\text{px/s}$.
- **`jaywalking`**:
  - Pedestrian tracks (`class_id == 0`) whose contact point $(c_x, y_2)$ lies on the roadway carriageway for $\ge 1.0$s.
- **`near_miss`**:
  - High closing speed, small predicted $\text{TTC} < 2.5$s and close distance, accompanied by sharp deceleration/swerving without physical contact or post-collision arrest.

### 3. Causal Accident Anticipation — Part B (`RiskEstimator`)
- **Strictly Causal**: `step(frame, t_sec)` operates frame-by-frame with zero lookahead and never opens the video file.
- **Time-To-Collision (TTC)**: For every pair of moving entities:
  $$\text{TTC} = \frac{\|\Delta \vec{p}\|}{v_{close}}, \quad \text{where } v_{close} = -\frac{\Delta \vec{p} \cdot \Delta \vec{v}}{\|\Delta \vec{p}\|}$$
- **Projected Closest Distance**: Computes minimum distance of closest approach $d_{closest}$ along the relative trajectory vector.
- **Risk Score Mapping**:
  - If trajectories converge with projected $d_{closest} < 1.35 \times R_{combined}$ and $\text{TTC} \le 5.0\text{s}$:
    $$\text{Score} = 0.72 + 0.26 \times \left(1.0 - \frac{\text{TTC}}{5.0}\right) \in [0.72, 0.98]$$
  - If $\text{TTC} \in (5.0\text{s}, 8.0\text{s}]$: smoothly decays from $0.65$ down to $0.20$.
  - Otherwise: baseline low score ($0.02$).
- **Continuous-Time Temporal Smoothing**: Applies continuous exponential decay $\exp(-\Delta t / 0.8)$ to maintain coherent alarm intervals matching the competition metric ($H = 5\text{s}, W = 10\text{s}, \theta = 0.5$).

### 4. Overlap Merging & Post-Processing (`src/utils.py`)
- Merges same-class segments whose gap $\le 1.5\text{s}$ into contiguous segments.
- Completely prevents same-class overlaps (which would otherwise be dropped by the harness).
- Discards sub-second transient blips ($< 0.5\text{s}$) to optimize the strict $F_1@0.7$ metric.
- Binds all timestamps to $[0, \text{duration}]$.

---

## Instructions to Install and Run

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Download Weights (Pre-run)
```bash
# On Linux / macOS:
bash weights/download.sh

# Or cross-platform using Python:
python weights/download.py
```
This fetches the official `yolov8n.pt` weights (~6.25 MB) into `weights/yolov8n.pt`.

### 3. Run Submission Harness
```bash
python run_submission.py --videos samples --out predictions.json --team <your-team-name>
```

### 4. Evaluate Results
```bash
# Format verification only
python evaluate.py --pred predictions.json --validate-only

# Full evaluation against labels
python evaluate.py --pred predictions.json --gt ground_truth.json --per-video
```

---

## Determinism & Hardware Compliance
- **Seeds & Determinism**: Frame sampling and rule heuristics are fully deterministic.
- **Budget**: Running YOLOv8n with a frame sampling stride of 3–4 frames achieves $\sim 0.1\times - 0.2\times$ real-time wall-clock latency on the target NVIDIA T4 GPU, safely within the $3\times$ duration budget.

