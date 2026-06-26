import math
from collections import deque

class SeizureDetector:
    def __init__(self, window_size=90, jerk_threshold=12, amplitude_min=2.0, amplitude_max=40.0):
        # 90 frames @ 30fps = 3 seconds of temporal memory
        self.window_size = window_size
        
        # How many directional reversals (jerks) in 3 seconds trigger a convulsion warning
        self.jerk_threshold = jerk_threshold 
        
        # Ignore microscopic camera noise (<2px) and massive macro movements (>40px/frame)
        self.amplitude_min = amplitude_min   
        self.amplitude_max = amplitude_max   

        # Track the extremities where convulsions are most visible
        self.history = {
            "l_wrist": deque(maxlen=window_size),
            "r_wrist": deque(maxlen=window_size),
            "l_ankle": deque(maxlen=window_size),
            "r_ankle": deque(maxlen=window_size)
        }

        self.is_convulsing = False

    def update(self, frame_data, fsm_state, immobility_data):
        # 1. Update positional history
        for key in self.history.keys():
            if "wrist" in key:
                idx = 0 if key == "l_wrist" else 1
                pt = frame_data.get("wrists", (None, None))[idx]
            else:
                idx = 0 if key == "l_ankle" else 1
                pt = frame_data.get("ankles", (None, None))[idx]

            if pt:
                self.history[key].append(pt)

        # 2. Calculate High-Frequency Oscillations (Jerks)
        max_jerks = 0
        for key, points in self.history.items():
            if len(points) < 10:
                continue

            jerks = 0
            for i in range(2, len(points)):
                p0, p1, p2 = points[i-2], points[i-1], points[i]

                # Velocity vectors for the last two frames
                v1_x, v1_y = p1[0] - p0[0], p1[1] - p0[1]
                v2_x, v2_y = p2[0] - p1[0], p2[1] - p1[1]

                # Vector magnitudes (Amplitudes)
                amp1 = math.hypot(v1_x, v1_y)
                amp2 = math.hypot(v2_x, v2_y)

                # Only count jerks that are within the convulsive amplitude range
                if self.amplitude_min < amp1 < self.amplitude_max and self.amplitude_min < amp2 < self.amplitude_max:
                    # Dot product calculation. A negative result means the angle between vectors is > 90 degrees (a sharp reversal)
                    dot_product = (v1_x * v2_x) + (v1_y * v2_y)
                    if dot_product < 0:
                        jerks += 1

            if jerks > max_jerks:
                max_jerks = jerks

        # 3. Fuse Data into Seizure Confidence Score
        confidence = 0.0
        alerts = []

        # Rule A: Convulsive Movements
        if max_jerks >= self.jerk_threshold:
            self.is_convulsing = True
            confidence += 0.50
            alerts.append(f"Convulsions Detected ({max_jerks} jerks/3s)")
        else:
            self.is_convulsing = False

        # Rule B: Collapse
        if fsm_state in ["GROUND_CONTACT", "NO_RECOVERY"]:
            confidence += 0.25

        # Rule C: Post-ictal Unresponsiveness
        is_immobile = immobility_data.get("is_immobile", False) if immobility_data else False
        if is_immobile and fsm_state == "NO_RECOVERY":
            confidence += 0.15
            
        # Ensure it never exceeds 1.0 (or 0.99 for realistic bounds)
        final_confidence = min(0.99, confidence)

        event = "none"
        if final_confidence >= 0.75:
            event = "possible seizure"
            alerts.insert(0, f"SEIZURE WARNING! (Conf: {final_confidence:.2f})")

        # Returns the exact JSON output requested in your spec
        return {
            "event": event,
            "confidence": round(final_confidence, 2),
            "alerts": alerts
        }