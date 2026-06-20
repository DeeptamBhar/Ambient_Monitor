import time
from collections import deque

class AgitationDetector:
    """
    Detects behavioral agitation and restlessness in patients.
    
    Monitors two key indicators of agitation:
    1. Posture shifts: Frequent transitions between sitting/standing/lying
    2. Pacing: Repeated back-and-forth movement (path reversals)
    
    Generates a 0-100 risk score and identifies alerts when thresholds exceeded.
    """
    
    def __init__(self, time_window=60.0, pacing_dist_px=50.0, posture_thresh=60.0):
        """
        Initialize the agitation detector with behavioral sensitivity parameters.
        
        Args:
            time_window (float, optional): Duration in seconds to track behavioral events.
                Events older than this window are discarded. Defaults to 60.0 (1 minute).
            pacing_dist_px (float, optional): Minimum horizontal distance (pixels) before counting
                movement in one direction. Prevents micro-jitter from registering as pacing. Defaults to 50.0.
            posture_thresh (float, optional): Body angle threshold (degrees) for detecting posture shifts.
                Used to differentiate sitting (<60°) from standing (>60°). Defaults to 60.0.
        """
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
        """
        Process a new frame to detect agitation indicators.
        
        Tracks posture shifts (stand/sit transitions) and pacing patterns (directional reversals).
        Automatically prunes events outside the time window and returns a risk assessment.
        
        Args:
            frame_data (dict): Frame containing joint positions, particularly 'hips' for tracking position.
            theta (float): Current body angle in degrees.
        
        Returns:
            dict: Risk assessment containing:
                'score' (int): 0-100 agitation score
                'risk' (str): 'low', 'medium', or 'high'
                'alerts' (list): List of specific alert messages if thresholds exceeded
        """
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
        """
        Remove behavioral events from history that are outside the configured time window.
        
        Maintains a sliding window of recent events by removing old entries from both
        posture_shifts and path_reversals queues.
        
        Args:
            current_time (float): Current timestamp in seconds (typically time.time()).
        
        Returns:
            None. Modifies internal event queues in-place.
        """
        while self.posture_shifts and current_time - self.posture_shifts[0] > self.time_window:
            self.posture_shifts.popleft()
        while self.path_reversals and current_time - self.path_reversals[0] > self.time_window:
            self.path_reversals.popleft()

    def _generate_score(self):
        """
        Calculate agitation risk score from behavioral event frequency.
        
        Implements weighted scoring:
        - Posture shifts (stand/sit transitions): 15 points each (highly indicative of restlessness)
        - Path reversals (pacing back-and-forth): 10 points each
        
        Maps score to risk level:
        - 0-29: Low risk
        - 30-69: Medium risk
        - 70-100: High risk
        
        Returns:
            dict: Containing 'score' (int 0-100), 'risk' (str), and 'alerts' (list of specific behaviors)
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