# Ambient Monitor

A medical-grade Ambient Intelligence Platform designed to monitor patient mobility and detect falls in real-time.

It extracts raw pose data via YOLOv8, computes human kinematics over a temporal rolling buffer, and passes the state vectors into a deterministic Finite State Machine (FSM). This modular architecture ensures that complex actions (e.g., doing burpees, sitting quickly, or dropping objects) are filtered out mathematically.

## System Architecture

The pipeline is structured similarly to an autonomous navigation stack, separated into distinct functional nodes:

1. **Perception Node (`core/perception.py`)**
   - Runs ultralytics YOLOv8-pose to extract 17-point COCO keypoints.
   - Isolates the head, neck, shoulders, hips, knees, and ankles.

2. **Temporal Memory (`core/kinematics.py`)**
   - Maintains a fixed-length rolling buffer (default 30 frames) to provide historical context.
   - Handles brief occlusion events by duplicating the last known valid state to prevent mathematical exceptions.

3. **Kinematics (`core/kinematics.py`)**
   - Calculates continuous real-time state variables:
     - **Vertical Velocity ($v_y$):** Evaluated via the rate of change of the mid-hip coordinate ($\Delta y / \Delta t$).
     - **Body Angle ($\theta$):** The vector angle between the mid-hip and the calculated neck coordinate relative to the horizontal axis.

4. **Logic (`core/fsm.py`)**
   - A deterministic Finite State Machine that evaluates the kinematic telemetry.
   - Tracks progression through strict physical states: `STANDING` $\rightarrow$ `LOSING_BALANCE` $\rightarrow$ `RAPID_DESCENT` $\rightarrow$ `GROUND_CONTACT` $\rightarrow$ `NO_RECOVERY`.
   - Classifies fall direction and generates automated JSON alert payloads.

## Directory Structure

```text
ambient_monitor/
├── data/
│   └── inputs/                 # Drop test .mp4 files here
├── core/                       
│   ├── __init__.py
│   ├── fsm.py                  
│   └── kinematics.py           
├── utils/
│   ├── __init__.py
│   └── visualizer.py           
├── config.yaml                  
├── main.py                     
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
python main.py
```

**To run a test video:**

```bash
python main.py --source data/inputs/burpees.mp4
```

##  Configuration (`config.yaml`)

All mathematical thresholds, buffer sizes, and FSM parameters are exposed in `config.yaml` for quick iteration and camera-angle tuning without modifying source code.

```yaml
kinematics:
  max_frames: 30       # Temporal memory window
  fps: 30.0

fsm:
  vy_fall_thresh: 150.0 # Velocity required to trigger free-fall (px/s)
  theta_imbalance: 60.0 # Angle deviation to trigger warning state (deg)
  max_recovery_time: 5.0 # Seconds on the floor before critical alert
```

