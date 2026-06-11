# Ambient Monitor

A medical-grade Ambient Intelligence Platform designed to monitor patient mobility, detect falls, and run live clinical gait diagnostics.
 
Unlike standard "black-box" end-to-end AI models, this system utilizes a decoupled perception-control stack. It extracts raw pose data via YOLOv8, computes full 2D kinematics and signal peaks over temporal rolling buffers, and passes state vectors into a deterministic Finite State Machine (FSM). This modular architecture is highly robust, mathematically debuggable, and immune to standard false positives (e.g., exercise or dropping objects).

## System Architecture

The pipeline is structured similarly to an autonomous navigation stack, separated into distinct functional nodes:

1. **Perception Node (`core/perception.py`)**
   - Runs ultralytics YOLOv8-pose to extract 17-point COCO keypoints.
   - Isolates critical joints: head, neck, shoulders, wrists, hips, knees, and ankles.

2. **Temporal Memory (`core/kinematics.py`)**
   - Maintains a fixed-length rolling buffer (30 frames) to provide historical context.
   - Handles brief occlusion events by duplicating the last known valid state to prevent pipeline crashes.

3. **Kinematics (`core/kinematics.py`)**
   Calculates continuous real-time state variables:
 
   - **Total Velocity ($v_{total}$):** Evaluates the full 2D magnitude of the hip coordinate's rate of change ($\sqrt{v_x^2 + v_y^2}$) to catch both vertical drops and horizontal projectiles.
   - **Body Angle ($\theta$):** The vector angle between the mid-hip and neck relative to the horizontal axis.

4. **Logic (`core/fsm.py`)**
   - A deterministic Finite State Machine evaluating kinematic telemetry.
   - Tracks strict physical states: `STANDING` → `LOSING_BALANCE` → `RAPID_DESCENT` → `GROUND_CONTACT` → `NO_RECOVERY`.
   - **Kinetic Override:** Bypasses standard recovery timers instantly for violent, high-velocity impacts.
   - Classifies fall biomechanics (Forward, Backward, Side).

5. **Gait Analyzer (`core/gait.py`)**
   - Runs live signal processing on ankle distance ($\Delta x$) to detect extension peaks.
   - Extracts **Stride Length** (pixels), **Cadence** (steps/min), and **Arm Swing** amplitude.
   - Evaluates telemetry against disease indicators to flag Parkinson's (shuffling gait), Stroke (asymmetry), and Frailty (low overall speed).

6. **Immobility Tracker (`core/immobility.py`)**
   - Long-term temporal stopwatch tracking macro-movements and posture shifts.
   - Flags severe unresponsiveness or prolonged static positioning based on a configurable noise floor.

7. **Agitation Detector (`core/agitation.py`)**
   - Translates behavioral patterns into a quantitative Agitation Score (0-100) over a sliding time window.
   - Tracks X-axis reversals for pacing and rapid angle boundary crossings for restlessness.

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
│   └── kinematics.py           # Calculus & temporal buffer
├── utils/
│   ├── __init__.py
│   └── visualizer.py           # Dynamic telemetry HUD
├── config.yaml                 # Centralized thresholds
├── main.py                     # Pipeline orchestrator
└── requirements.txt
```

## Quick Start

### 1. Environment Setup
Run this in a clean virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

The system can process live webcam feeds or pre-recorded video files.

**To run via Webcam:**

```bash
python3 main.py
```

**To run a test video:**

```bash
python3 main.py --source data/inputs/burpees.mp4
```



