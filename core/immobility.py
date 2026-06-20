import time

class ImmobilityTracker:
    """
    Tracks prolonged immobility and triggers alerts if patient remains motionless too long.
    
    Monitors two independent stopwatches: spatial movement and postural changes.
    Immobility is confirmed when both indicators show no activity for the critical threshold.
    Uses the minimum of the two timers to ensure any motion or position change resets the alert.
    """
    
    def __init__(self, movement_thresh_px=5.0, posture_thresh_deg=15.0, critical_time_sec=2700.0):
        """
        Initialize the immobility tracker with sensitivity thresholds.
        
        Args:
            movement_thresh_px (float, optional): Minimum horizontal displacement (pixels)
                to register as "movement". Below this threshold, motion is considered noise/jitter.
                Defaults to 5.0.
            posture_thresh_deg (float, optional): Minimum body angle change (degrees)
                to register as "posture shift". Defaults to 15.0.
            critical_time_sec (float, optional): Duration (seconds) of complete motionlessness
                before triggering immobility alert. Defaults to 2700.0 (45 minutes).
                For bed-bound patients, this prevents false alarms during sleep.
        """
        # The "Noise Floor" - Ignore micro-jitters
        self.movement_thresh = movement_thresh_px
        self.posture_thresh = posture_thresh_deg
        
        # How long before an alert is triggered (e.g., 2700s = 45 mins)
        self.critical_time_sec = critical_time_sec 
        
        # Stopwatches
        self.last_movement_time = time.time()
        self.last_posture_time = time.time()
        self.last_theta = None
        
        self.is_immobile = False

    def update(self, v_total, theta):
        """
        Updates the immobility stopwatches based on current kinematics.
        """
        current_time = time.time()
        
        # 1. Check for spatial movement (Did they move across the bed/room?)
        if v_total > self.movement_thresh:
            self.last_movement_time = current_time
            self.is_immobile = False
            
        # 2. Check for posture shifts (Did they roll over or sit up?)
        if self.last_theta is not None:
            if abs(theta - self.last_theta) > self.posture_thresh:
                self.last_posture_time = current_time
                self.is_immobile = False # Posture shift counts as breaking immobility
        self.last_theta = theta
        
        # Calculate how long they have been completely frozen
        motionless_duration = current_time - self.last_movement_time
        posture_static_duration = current_time - self.last_posture_time
        
        # Determine the "True" motionless time (the lower of the two timers)
        true_immobile_time = min(motionless_duration, posture_static_duration)
        
        alerts = []
        if true_immobile_time > self.critical_time_sec:
            self.is_immobile = True
            # Convert to minutes for the alert payload
            mins = int(true_immobile_time // 60)
            alerts.append(f"CRITICAL: Patient motionless for {mins} minutes.")
            
        return {
            "motionless_sec": round(true_immobile_time, 1),
            "is_immobile": self.is_immobile,
            "alerts": alerts
        }