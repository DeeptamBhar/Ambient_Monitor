# Ambient Monitor

A medical-grade Ambient Intelligence Platform designed to monitor patient mobility, detect falls, and run live clinical diagnostics.
 
Unlike standard "black-box" end-to-end AI models, this system utilizes a decoupled, dual-model perception-control stack. It extracts raw pose data and environmental objects via YOLO11, computes full 2D kinematics and signal peaks over temporal rolling buffers, and passes state vectors into a deterministic Finite State Machine (FSM). This modular architecture is highly robust, mathematically debuggable, and immune to standard false positives (e.g., exercise, dropping objects, or non-target personnel walking into the room).

## System Architecture

The pipeline is structured similarly to an autonomous navigation stack, separated into distinct functional nodes:

1. **Perception Node (Dual-Model Sensor Fusion)**
   - **Pose Tracking (`yolo11n-pose.pt`):** Extracts 17-point COCO keypoints with persistent ID tracking across frames. Includes an Edge Rejection Filter to dynamically lock onto resident patients and ignore transient personnel (like nurses) walking through doorways.
   - **Object Detection (`hospital_weights.pt`):** A custom-trained YOLO model that detects specific medical commodities (e.g., walkers, wheelchairs, beds) to establish environmental context.

2. **Temporal Memory (`core/kinematics.py`)**
   - Maintains a fixed-length rolling buffer (30 frames) to provide historical context.
   - Handles brief occlusion events by duplicating the last known valid state to prevent pipeline crashes.

3. **Kinematics (`core/kinematics.py`)**
   Calculates continuous real-time state variables:
   - **Total Velocity ($v_{total}$):** Evaluates the full 2D magnitude of the hip coordinate's rate of change ($\sqrt{v_x^2 + v_y^2}$) to catch both vertical drops and horizontal projectiles.
   - **Body Angle ($\theta$):** The vector angle between the mid-hip and neck relative to the horizontal axis.
   - **Z-Axis Foreshortening:** Uses proportional thigh-to-torso ratios to detect frontal sitting versus standing.

4. **Logic (`core/fsm.py`)**
   - A deterministic Finite State Machine evaluating kinematic telemetry.
   - Tracks strict physical states: `STANDING` → `LOSING_BALANCE` → `RAPID_DESCENT` → `GROUND_CONTACT` → `NO_RECOVERY`.
   - **Kinetic Override:** Bypasses standard recovery timers instantly for violent, high-velocity impacts.
   - Classifies fall biomechanics (Forward, Backward, Side).

5. **Gait Analyzer (`core/gait.py`)**
   - Runs live signal processing on ankle distance ($\Delta x$) to detect extension peaks.
   - Extracts Stride Length (pixels), Cadence (steps/min), and Arm Swing amplitude.
   - Evaluates telemetry against disease indicators to flag Parkinson's (shuffling gait), Stroke (asymmetry), and Frailty.

6. **Immobility Tracker (`core/immobility.py`)**
   - Long-term temporal stopwatch tracking macro-movements and posture shifts.
   - Flags severe unresponsiveness or prolonged static positioning based on a configurable noise floor.

7. **Agitation Detector (`core/agitation.py`)**
   - Translates behavioral patterns into a quantitative Agitation Score (0-100) over a sliding time window.
   - Tracks X-axis reversals for pacing and rapid angle boundary crossings for restlessness.

8. **Wandering Detector (`core/wandering.py`)**
   - Maps spatial zones within the room and tracks pacing behaviors over extended periods.
   - Adjusts risk profiles dynamically based on time-of-day (circadian/sundowning context).

9. **Seizure Detector (`core/seizure.py`)**
   - Evaluates high-frequency oscillation (jerks) using vector dot-products on wrist and ankle trajectories over a 3-second sliding window.
   - Fuses convulsion data with FSM collapse states and post-ictal immobility to generate a seizure confidence score.

10. **Safety Monitor (`core/safety.py`)**
    - **Physical Support Logic:** Uses mathematical intersection (Bounding Box IoU) between the patient's pose and detected environmental objects (e.g., flagging unassisted mobility if a fall-risk patient abandons their walker).
    - **Behavioral Proxies:** Utilizes kinematic heuristics (e.g., hands persistently near the neck/face) to detect clear-plastic IV line and oxygen tube pulling.

## Directory Structure

```text
ambient_monitor/
├── data/
│   └── inputs/                 # Drop test .mp4 files here
├── core/                       
│   ├── __init__.py
│   ├── agitation.py            # Behavioral scoring & pacing logic
│   ├── fsm.py                  # State machine & payload generation
│   ├── gait.py                 # Signal processing & diagnostics
│   ├── immobility.py           # Long-term static monitoring
│   ├── kinematics.py           # Calculus, temporal buffer & Z-axis limits
│   ├── safety.py               # Sensor fusion & behavioral heuristics
│   ├── seizure.py              # High-frequency vector oscillation
│   └── wandering.py            # Spatial zone mapping & sundowning
├── utils/
│   ├── __init__.py
│   └── visualizer.py           # Dynamic telemetry HUD & Object BBoxes
├── config.yaml                 # Centralized thresholds & Patient Profiles
├── main.py                     # Dual-Model Pipeline orchestrator
├── hospital_weights.pt         # Custom YOLO model for medical commodities
└── requirements.txt
```

## Quick Start

### 1. Environment Setup
Run this in a clean virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

The system can process live webcam feeds or pre-recorded video files. It will automatically download the standard yolo11n-pose.pt base model on first execution.

**To run via Webcam:**

```bash
python3 main.py
```

**To run a test video:**

```bash
python3 main.py --source data/inputs/burpees.mp4
```



