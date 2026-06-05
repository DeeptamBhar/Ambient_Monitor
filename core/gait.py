import time
import numpy as np
from collections import deque

class GaitAnalyzer:
    def __init__(self, fps=30.0, step_cooldown=0.5):
        self.fps = fps
        self.step_cooldown = step_cooldown # Minimum seconds between steps to prevent double-counting
        
        # Signal processing buffers
        self.ankle_dist_history = deque(maxlen=15) # Short window for peak detection
        
        # Gait Metrics
        self.step_timestamps = deque(maxlen=10) # Store times of the last 10 steps
        self.current_stride_length = 0.0        # In pixels
        self.cadence = 0.0                      # Steps per minute

    def update(self, frame_data):
        """
        Ingests frame data to calculate Stride Length and Step Frequency.
        Returns a dictionary of current gait metrics.
        """
        l_ankle, r_ankle = frame_data.get("ankles", (None, None))
        
        if not l_ankle or not r_ankle:
            return self._get_metrics()

        # Calculate absolute horizontal distance between ankles
        dx_ankles = abs(l_ankle[0] - r_ankle[0])
        self.ankle_dist_history.append(dx_ankles)

        # We need at least 3 frames to detect a peak (a value higher than its neighbors)
        if len(self.ankle_dist_history) >= 3:
            # Check if the middle value in our recent history is a peak
            p0 = self.ankle_dist_history[-3]
            p1 = self.ankle_dist_history[-2] # Potential peak
            p2 = self.ankle_dist_history[-1]

            if p1 > p0 and p1 > p2:
                # We found a stride extension
                current_time = time.time()
                
                # Check cooldown to avoid microscopic jitter triggering a step
                if not self.step_timestamps or (current_time - self.step_timestamps[-1]) > self.step_cooldown:
                    self.step_timestamps.append(current_time)
                    self.current_stride_length = p1
                    self._calculate_cadence()

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

    def _get_metrics(self):
        return {
            "stride_length_px": round(self.current_stride_length, 2),
            "cadence_spm": round(self.cadence, 1)
        }