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
                 v_total_fall_thresh=150.0,   # Swapped vy for v_total
                 critical_impact_thresh=350.0, # The "Bike Crash" threshold
                 theta_horizontal=30.0, 
                 vy_impact=50.0, 
                 max_recovery_time=5.0):
        
        self.state = FallState.STANDING
        
        self.theta_imbalance = theta_imbalance
        self.v_total_fall_thresh = v_total_fall_thresh
        self.critical_impact_thresh = critical_impact_thresh
        self.theta_horizontal = theta_horizontal
        self.vy_impact = vy_impact
        self.max_recovery_time = max_recovery_time
        
        self.ground_contact_time = 0.0
        self.kinetic_override = False # Tracks if the crash was severe

    def update(self, v_total, vy, theta):
        if self.state == FallState.STANDING:
            if theta < self.theta_imbalance:
                self.state = FallState.LOSING_BALANCE
                self.kinetic_override = False # Reset flag

        elif self.state == FallState.LOSING_BALANCE:
            if v_total > self.critical_impact_thresh:
                # Extreme horizontal or vertical impact
                self.kinetic_override = True
                self.state = FallState.RAPID_DESCENT
            elif v_total > self.v_total_fall_thresh:
                # Standard fall
                self.state = FallState.RAPID_DESCENT
            elif theta >= self.theta_imbalance:
                self.state = FallState.STANDING

        elif self.state == FallState.RAPID_DESCENT:
            if vy < self.vy_impact and theta < self.theta_horizontal:
                self.state = FallState.GROUND_CONTACT
                self.ground_contact_time = time.time()
            elif theta >= self.theta_imbalance:
                self.state = FallState.STANDING

        elif self.state == FallState.GROUND_CONTACT:
            if self.kinetic_override:
                # BYPASS THE TIMER: The impact was too violent to wait 5 seconds.
                self.state = FallState.NO_RECOVERY
            elif theta >= self.theta_imbalance:
                self.state = FallState.STANDING
            elif (time.time() - self.ground_contact_time) > self.max_recovery_time:
                self.state = FallState.NO_RECOVERY

        elif self.state == FallState.NO_RECOVERY:
            if theta >= self.theta_imbalance:
                self.state = FallState.STANDING

        return self.state.value
    
    def generate_alert_payload(self, frame_data, confidence=0.94):
        """
        Generates the severity score and JSON output matching the spec doc.
        Uses geometric heuristics to classify Forward, Backward, and Side falls.
        """
        # if self.state != FallState.NO_RECOVERY:
        #     return None

        time_on_ground = round(time.time() - self.ground_contact_time, 1)
        
        l_shoulder, r_shoulder = frame_data.get("shoulders", (None, None))
        head = frame_data.get("head") # Nose keypoint
        
        classification = "Unknown"
        head_impact = "unknown"
        severity = "high"

        if l_shoulder and r_shoulder:
            # Calculate the horizontal and vertical spread of the shoulders
            shoulder_dx = abs(l_shoulder[0] - r_shoulder[0])
            shoulder_dy = abs(l_shoulder[1] - r_shoulder[1])

            # If shoulders are stacked vertically or pinched horizontally -> Side Fall
            if shoulder_dx < 40 or shoulder_dy > shoulder_dx:
                classification = "Side Fall"
                head_impact = "unlikely"
            else:
                # If nose is pitched way below the shoulders -> Forward Fall
                if head and head[1] > min(l_shoulder[1], r_shoulder[1]) + 20:
                    classification = "Forward Fall"
                    head_impact = "possible"
                    severity = "critical" # Flag critical for potential facial trauma
                else:
                    classification = "Backward Fall"
                    head_impact = "possible" # Flag possible for spine/back of head impact

        # Time override: If they are on the floor for 2 minutes, it's critical regardless
        if time_on_ground > 120:
            severity = "critical"

        # The exact JSON structure 
        payload = {
            "event": "fall",
            "classification": classification,
            "severity": severity,
            "confidence": confidence,
            "head_impact": head_impact,
            "time_on_ground": f"{time_on_ground} sec"
        }
        
        return payload
