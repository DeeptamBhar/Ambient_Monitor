import time
from collections import deque

class AgitationDetector:
    def __init__(self, time_window=60.0, pacing_dist_px=50.0, posture_thresh=60.0):
        # How long our memory window is (default: look at the last 60 seconds)
        self.time_window = time_window 
        
        # Minimum pixel distance moved before we log a "direction" for pacing
        self.pacing_dist_px = pacing_dist_px
        self.posture_thresh = posture_thresh
        
        # Rolling event logs (stores timestamps of events)
        self.posture_shifts = deque()
        self.path_reversals = deque()
        
        # State tracking
        self.last_theta = None
        self.last_x = None
        self.current_direction = 0 # 1 for right, -1 for left
        self.accumulated_x = 0.0

    def update(self, frame_data, theta):
        current_time = time.time()
        
        # 1. Clean old events out of our sliding window
        self._prune_history(current_time)

        # 2. Track Restlessness (Constant Stand/Sit transitions)
        if self.last_theta is not None:
            # If they cross the 60 degree boundary, log a shift
            if (self.last_theta > self.posture_thresh and theta <= self.posture_thresh) or \
               (self.last_theta <= self.posture_thresh and theta > self.posture_thresh):
                self.posture_shifts.append(current_time)
        self.last_theta = theta

        # 3. Track Pacing (X-Axis Reversals)
        hips = frame_data.get("hips", (None, None))
        if hips[0] and hips[1]:
            mid_x = (hips[0][0] + hips[1][0]) / 2
            
            if self.last_x is not None:
                dx = mid_x - self.last_x
                self.accumulated_x += dx
                
                # Have they moved far enough in one direction to establish a vector?
                if abs(self.accumulated_x) > self.pacing_dist_px:
                    new_dir = 1 if self.accumulated_x > 0 else -1
                    
                    # If the vector changed (they turned around), log a reversal
                    if self.current_direction != 0 and new_dir != self.current_direction:
                        self.path_reversals.append(current_time)
                    
                    self.current_direction = new_dir
                    self.accumulated_x = 0.0 # Reset for the next leg of pacing
            self.last_x = mid_x

        return self._generate_score()

    def _prune_history(self, current_time):
        while self.posture_shifts and current_time - self.posture_shifts[0] > self.time_window:
            self.posture_shifts.popleft()
        while self.path_reversals and current_time - self.path_reversals[0] > self.time_window:
            self.path_reversals.popleft()

    def _generate_score(self):
        """
        Calculates a 0-100 score based on event frequency.
        """
        # Weightings: Posture shifts are highly indicative of agitation (15 pts each)
        # Reversals (pacing laps) are 10 pts each.
        score = (len(self.posture_shifts) * 15) + (len(self.path_reversals) * 10)
        score = min(100, int(score))
        
        risk = "low"
        if score >= 70:
            risk = "high"
        elif score >= 30:
            risk = "medium"
            
        alerts = []
        if len(self.posture_shifts) >= 3:
            alerts.append("Restlessness: Constant Posture Shifts")
        if len(self.path_reversals) >= 4:
            alerts.append("Pacing: Repeated Path Reversals")

        return {
            "score": score,
            "risk": risk,
            "alerts": alerts
        }