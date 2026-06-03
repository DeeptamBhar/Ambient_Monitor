import time
from enum import Enum

class FallState(Enum):
    STANDING = "STANDING"
    LOSING_BALANCE = "LOSING_BALANCE"
    RAPID_DESCENT = "RAPID_DESCENT"
    GROUND_CONTACT = "GROUND_CONTACT"
    NO_RECOVERY = "NO_RECOVERY"

class FallDetectorFSM:
    def __init__(self, 
                 theta_imbalance=60.0, 
                 vy_fall_thresh=150.0, 
                 theta_horizontal=30.0, 
                 vy_impact=50.0, 
                 max_recovery_time=5.0):
        
        # Initial state
        self.state = FallState.STANDING
        
        # The thresholds (you will tune these later based on your camera angle)
        self.theta_imbalance = theta_imbalance      # Degrees
        self.vy_fall_thresh = vy_fall_thresh        # Pixels per second
        self.theta_horizontal = theta_horizontal    # Degrees
        self.vy_impact = vy_impact                  # Pixels per second
        self.max_recovery_time = max_recovery_time  # Seconds
        
        self.ground_contact_time = 0.0

    def update(self, vy, theta):
        """
        Takes the current vertical velocity and body angle and shifts the FSM state.
        Returns the current state.
        """
        
        # 1. STANDING: Safe state. Waiting for a sign of imbalance.
        if self.state == FallState.STANDING:
            if theta < self.theta_imbalance:
                self.state = FallState.LOSING_BALANCE

        # 2. LOSING BALANCE: Person is tilted. Are they falling or just bending over?
        elif self.state == FallState.LOSING_BALANCE:
            if vy > self.vy_fall_thresh:
                # High velocity + tilt = Falling
                self.state = FallState.RAPID_DESCENT
            elif theta >= self.theta_imbalance:
                # They stood back up. False alarm.
                self.state = FallState.STANDING

        # 3. RAPID DESCENT: They are in free-fall. Waiting for impact.
        elif self.state == FallState.RAPID_DESCENT:
            # Impact means velocity drops near zero, and they are horizontal
            if vy < self.vy_impact and theta < self.theta_horizontal:
                self.state = FallState.GROUND_CONTACT
                self.ground_contact_time = time.time() # Start the timer
            elif theta >= self.theta_imbalance:
                # Somehow recovered mid-air or false positive
                self.state = FallState.STANDING

        # 4. GROUND CONTACT: They hit the floor. Are they doing a sit-up, or are they hurt?
        elif self.state == FallState.GROUND_CONTACT:
            if theta >= self.theta_imbalance:
                # They stood up. False alarm killed.
                self.state = FallState.STANDING
            elif (time.time() - self.ground_contact_time) > self.max_recovery_time:
                # They've been on the ground too long. Trigger the alarm.
                self.state = FallState.NO_RECOVERY

        # 5. NO RECOVERY: The alarm state. 
        elif self.state == FallState.NO_RECOVERY:
            # If they eventually stand up, reset the system.
            if theta >= self.theta_imbalance:
                self.state = FallState.STANDING

        return self.state.value
    
    def generate_alert_payload(self, frame_data, confidence=0.94):
        """
        Generates the severity score and JSON output required by the spec.
        """
        if self.state != FallState.NO_RECOVERY:
            return None

        time_on_ground = round(time.time() - self.ground_contact_time, 1)
        
        # Basic heuristic for classification: 
        # Compare nose (head) position to shoulders to guess the fall direction.
        # In a real 3D system, you'd use depth, but for 2D we estimate based on visibility.
        head = frame_data.get("head")
        shoulders = frame_data.get("shoulders", (None, None))
        
        classification = "unknown"
        head_impact = "unknown"
        
        if head and shoulders[0] and shoulders[1]:
            # If the head is visible and below the shoulders, they likely pitched forward or sideways
            if head[1] > shoulders[0][1] and head[1] > shoulders[1][1]:
                classification = "Forward/Side Fall"
                head_impact = "possible"
            else:
                classification = "Backward Fall"
                head_impact = "unlikely"

        # Calculate Severity based on time on ground and head impact
        severity = "high"
        if time_on_ground > 120 or head_impact == "possible":
            severity = "critical"

        # The exact JSON structure requested in the design document
        payload = {
            "event": "fall",
            "classification": classification,
            "severity": severity,
            "confidence": confidence,
            "head impact": head_impact,
            "time_on_ground": f"{time_on_ground} sec"
        }
        
        return payload