import cv2

class DebugVisualizer:
    """
    Renders real-time telemetry, diagnostic data, and alerts onto video frames.
    
    Provides visual feedback for:
    - Pose skeleton and detection confidence
    - Kinematic measurements (velocity, body angle)
    - FSM state and fall classification
    - Clinical diagnostics (gait metrics, agitation, immobility, wandering alerts)
    - Critical fall alerts with dynamic visual emphasis
    """
    
    def __init__(self):
        """
        Initialize the debug visualizer with OpenCV rendering parameters.
        
        Sets up color palette (BGR format for OpenCV) and font parameters for rendering
        text overlays on video frames.
        """
        # OpenCV uses BGR (Blue, Green, Red) instead of RGB.
        self.colors = {
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "black": (0, 0, 0)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_yolo_skeleton(self, results):
        """
        Uses Ultralytics' built-in annotator to draw the bounding boxes and skeleton.
        Returns the annotated frame.
        """
        return results[0].plot()

    def draw_telemetry(self, frame, v_total, theta, current_state, buffer_size, classification="N/A", gait_metrics=None, immobility_data=None, agitation_data=None, wandering_data=None):
        """
        Render comprehensive telemetry and diagnostic data onto video frame.
        
        Overlays kinematics (velocity, body angle), FSM state, fall classification,
        and all clinical module diagnostics. Dynamically expands display area if
        multiple alerts are triggered to prevent text overflow.
        
        Args:
            frame (numpy.ndarray): Video frame (BGR format)
            v_total (float): Total velocity in pixels/second
            theta (float): Body angle in degrees
            current_state (str): Current FSM state ("STANDING", "RAPID_DESCENT", etc.)
            buffer_size (int): Current size of kinematic buffer (0-30 typically)
            classification (str, optional): Fall type classification ("Forward", "Backward", "Side", etc.)
            gait_metrics (dict, optional): Dictionary with stride_length_px, cadence_spm, speed_px_s, diagnostics
            immobility_data (dict, optional): Dictionary with motionless_sec, is_immobile, alerts
            agitation_data (dict, optional): Dictionary with score, risk, alerts
            wandering_data (dict, optional): Dictionary with risk, current_zone, is_night, alerts
        
        Returns:
            numpy.ndarray: Frame with rendered telemetry overlay
        """
        overlay = frame.copy()
        
        # Combine alerts from all active medical modules
        diagnostics = []
        if gait_metrics and "diagnostics" in gait_metrics:
            diagnostics.extend(gait_metrics["diagnostics"])
        if immobility_data and "alerts" in immobility_data:
            diagnostics.extend(immobility_data["alerts"])
        if agitation_data and "alerts" in agitation_data:
            diagnostics.extend(agitation_data["alerts"])
        if wandering_data and "alerts" in wandering_data: 
            diagnostics.extend(wandering_data["alerts"])

        # Base dimensions
        box_height = 290
        box_width = 350

        # Expand box dynamically if alerts are triggered
        if len(diagnostics) > 0:
            box_height += len(diagnostics) * 30
            box_width = 480

        # Draw the dynamic background box
        cv2.rectangle(overlay, (5, 5), (box_width, box_height), self.colors["black"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Print Buffer Status
        buffer_color = self.colors["green"] if buffer_size >= 10 else self.colors["yellow"]
        cv2.putText(frame, f"Buffer: {buffer_size}/30", (15, 30), 
                    self.font, 0.6, buffer_color, 2)

        # Print Kinematics
        cv2.putText(frame, f"v_total: {v_total:.2f} px/s", (15, 60), 
                    self.font, 0.6, self.colors["white"], 2)
        cv2.putText(frame, f"Theta:   {theta:.1f} deg", (15, 90), 
                    self.font, 0.6, self.colors["white"], 2)

        # Print FSM State
        state_color = self.colors["red"] if current_state in ["RAPID_DESCENT", "NO_RECOVERY", "GROUND_CONTACT"] else self.colors["green"]
        cv2.putText(frame, f"STATE: {current_state}", (15, 120), 
                    self.font, 0.7, state_color, 2)

        # Print Live Fall Classification
        class_color = self.colors["yellow"] if classification != "N/A" else self.colors["white"]
        cv2.putText(frame, f"TYPE:  {classification}", (15, 150), 
                    self.font, 0.7, class_color, 2)

        # Print Gait Metrics & Diagnostics
        if gait_metrics:
            stride = gait_metrics.get("stride_length_px", 0.0)
            cadence = gait_metrics.get("cadence_spm", 0.0)
            cv2.putText(frame, f"Stride:  {stride} px", (15, 190), 
                        self.font, 0.6, self.colors["white"], 2)
            cv2.putText(frame, f"Cadence: {cadence} spm", (15, 220), 
                        self.font, 0.6, self.colors["white"], 2)

        # Print Immobility Stopwatch
        if immobility_data:
            timer = immobility_data.get("motionless_sec", 0.0)
            is_immobile = immobility_data.get("is_immobile", False)
            
            # Format raw seconds into a clean MM:SS format
            mins = int(timer // 60)
            secs = int(timer % 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            timer_color = self.colors["red"] if is_immobile else self.colors["white"]
            cv2.putText(frame, f"Motionless: {time_str}", (15, 250), 
                        self.font, 0.6, timer_color, 2)
        
        # Agitation Score Output
        if agitation_data:
            score = agitation_data.get("score", 0)
            risk = agitation_data.get("risk", "low")
            
            # Color code the score
            score_color = self.colors["white"]
            if risk == "high": score_color = self.colors["red"]
            elif risk == "medium": score_color = self.colors["yellow"]
            
            cv2.putText(frame, f"Agitation: [{score}] Risk: {risk.upper()}", (15, 280), self.font, 0.6, score_color, 2)
        
        # Render Clinical Alerts dynamically at the bottom
        y_offset = 290
        for alert in diagnostics:
            cv2.putText(frame, f"! {alert}", (15, y_offset), self.font, 0.6, self.colors["red"], 2)
            y_offset += 30

        return frame

    def draw_critical_alert(self, frame, alert_payload):
        """
        Render full-screen critical alert for detected falls.
        
        Displays prominent red border and centered warning text indicating fall detection.
        Used when FSM enters NO_RECOVERY state to immediately notify caregivers.
        
        Args:
            frame (numpy.ndarray): Video frame (BGR format)
            alert_payload (dict): Alert data containing 'classification' (fall type) and severity info
        
        Returns:
            numpy.ndarray: Frame with red border and critical alert text overlay
        """
        if not alert_payload:
            return frame
            
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, 0), (w, h), self.colors["red"], 15)
        
        # Center warning text
        text = f"CRITICAL: {alert_payload['classification'].upper()}"
        text_size = cv2.getTextSize(text, self.font, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(frame, text, (text_x, h - 50), 
                    self.font, 1.2, self.colors["red"], 3)
                    
        return frame
