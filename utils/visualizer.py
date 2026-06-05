import cv2

class DebugVisualizer:
    def __init__(self):
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

    def draw_telemetry(self, frame, v_total, theta, current_state, buffer_size, classification="N/A", gait_metrics=None):
        """
        Overlays the kinematics data, FSM state, Fall Type, and live Gait Diagnostics.
        Dynamically resizes the background box if clinical alerts are triggered.
        """
        overlay = frame.copy()
        
        # Determine how many alerts we have to calculate box size
        diagnostics = []
        box_height = 230
        box_width = 350
        
        if gait_metrics and "diagnostics" in gait_metrics:
            diagnostics = gait_metrics["diagnostics"]
            if len(diagnostics) > 0:
                box_height += len(diagnostics) * 30
                box_width = 450 # Make it wider to fit long medical strings

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
            
            # Render Clinical Alerts dynamically
            y_offset = 250
            for alert in diagnostics:
                cv2.putText(frame, f"! {alert}", (15, y_offset), 
                            self.font, 0.6, self.colors["red"], 2)
                y_offset += 30

        return frame

    def draw_critical_alert(self, frame, alert_payload):
        """
        Flashes a massive red warning on the screen if a fall is detected.
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
