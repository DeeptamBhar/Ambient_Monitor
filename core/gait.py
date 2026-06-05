import time
import numpy as np
from collections import deque

class GaitAnalyzer:
    def __init__(self, fps=30.0, step_cooldown=0.5):
        self.fps = fps
        self.step_cooldown = step_cooldown
        
        # Signal processing buffers
        self.ankle_dist_history = deque(maxlen=15) 
        self.step_timestamps = deque(maxlen=10)
        
        # Diagnostic Buffers (Left vs Right)
        self.left_strides = deque(maxlen=5)
        self.right_strides = deque(maxlen=5)
        self.arm_swings = deque(maxlen=15)
        
        # Current Metrics
        self.current_stride_length = 0.0        
        self.cadence = 0.0                      
        self.walking_speed = 0.0 # Approximated in pixels/sec
        
        # Biomarker Thresholds (Can be moved to config.yaml later)
        self.thresh_shuffling_stride = 40.0 # pixels
        self.thresh_asymmetry = 0.75 # Ratio of short step to long step
        self.thresh_frailty_speed = 50.0 # pixels/sec

    def update(self, frame_data, v_total):
        """
        Ingests frame data and calculates clinical gait metrics.
        """
        self.walking_speed = v_total
        
        l_ankle, r_ankle = frame_data.get("ankles", (None, None))
        l_wrist, r_wrist = frame_data.get("wrists", (None, None)) # Assuming you extract wrists in main.py
        hips = frame_data.get("hips", (None, None))

        # 1. Track Arm Swing (Average distance of wrists to center hip)
        if l_wrist and r_wrist and hips[0] and hips[1]:
            mid_hip = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)
            l_swing = abs(l_wrist[0] - mid_hip[0])
            r_swing = abs(r_wrist[0] - mid_hip[0])
            self.arm_swings.append((l_swing + r_swing) / 2)

        # 2. Track Stride and Step Frequency
        if not l_ankle or not r_ankle:
            return self._get_metrics()

        dx_ankles = abs(l_ankle[0] - r_ankle[0])
        self.ankle_dist_history.append(dx_ankles)

        # Peak detection for steps
        if len(self.ankle_dist_history) >= 3:
            p0 = self.ankle_dist_history[-3]
            p1 = self.ankle_dist_history[-2] 
            p2 = self.ankle_dist_history[-1]

            if p1 > p0 and p1 > p2:
                current_time = time.time()
                
                if not self.step_timestamps or (current_time - self.step_timestamps[-1]) > self.step_cooldown:
                    self.step_timestamps.append(current_time)
                    self.current_stride_length = p1
                    self._calculate_cadence()
                    
                    # Estimate Left vs Right step by checking which ankle is further forward (X-axis)
                    if l_ankle[0] > r_ankle[0]:
                        self.left_strides.append(p1)
                    else:
                        self.right_strides.append(p1)

        return self._get_metrics()

    def _calculate_cadence(self):
        """
        Calculates steps per minute based on recent step timestamps.
        """
        if len(self.step_timestamps) < 2:
            self.cadence = 0.0
            return

        # Calculate average time between recent steps
        intervals = []
        for i in range(1, len(self.step_timestamps)):
            intervals.append(self.step_timestamps[i] - self.step_timestamps[i-1])
            
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval > 0:
            # Convert interval (seconds/step) to Cadence (steps/minute)
            self.cadence = 60.0 / avg_interval

    def diagnose(self):
        """
        Evaluates current metrics against clinical disease indicators.
        """
        alerts = []
        
        # 1. Parkinson's Indicators
        avg_swing = sum(self.arm_swings) / len(self.arm_swings) if self.arm_swings else 100.0
        if self.current_stride_length > 0 and self.current_stride_length < self.thresh_shuffling_stride and self.cadence > 40:
            alerts.append("Shuffling Gait Detected (Parkinson's Indicator)")
        if avg_swing < 20.0: # Minimal distance from wrists to body
            alerts.append("Reduced Arm Swing")

        # 2. Stroke Indicators (Asymmetry)
        if len(self.left_strides) >= 2 and len(self.right_strides) >= 2:
            avg_left = sum(self.left_strides) / len(self.left_strides)
            avg_right = sum(self.right_strides) / len(self.right_strides)
            symmetry_ratio = min(avg_left, avg_right) / max(avg_left, avg_right) if max(avg_left, avg_right) > 0 else 1.0
            
            if symmetry_ratio < self.thresh_asymmetry:
                alerts.append(f"Asymmetric Gait: {symmetry_ratio:.2f} (Stroke Indicator)")

        # 3. Frailty Indicators
        if 0 < self.walking_speed < self.thresh_frailty_speed:
            alerts.append("Slow Walking Speed (Frailty Indicator)")

        return alerts

    def _get_metrics(self):
        return {
            "stride_length_px": round(self.current_stride_length, 2),
            "cadence_spm": round(self.cadence, 1),
            "speed_px_s": round(self.walking_speed, 2),
            "diagnostics": self.diagnose()
        }