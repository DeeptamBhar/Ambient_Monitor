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

    def draw_telemetry(self, frame, vy, theta, current_state, buffer_size):
        """
        Overlays the kinematics data and FSM state directly onto the frame.
        """
        # 1. Draw a semi-transparent background box for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 140), self.colors["black"], -1)
        # Blend the overlay with the original frame (opacity = 0.6)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 2. Print Buffer Status
        buffer_color = self.colors["green"] if buffer_size >= 10 else self.colors["yellow"]
        cv2.putText(frame, f"Buffer: {buffer_size}/30", (15, 30), 
                    self.font, 0.6, buffer_color, 2)

        # 3. Print Kinematics (Math)
        cv2.putText(frame, f"v_y:   {vy:.2f} px/s", (15, 60), 
                    self.font, 0.6, self.colors["white"], 2)
        cv2.putText(frame, f"Theta: {theta:.1f} deg", (15, 90), 
                    self.font, 0.6, self.colors["white"], 2)

        # 4. Print FSM State
        state_color = self.colors["red"] if current_state in ["RAPID_DESCENT", "NO_RECOVERY"] else self.colors["green"]
        cv2.putText(frame, f"STATE: {current_state}", (15, 120), 
                    self.font, 0.7, state_color, 2)

        return frame

    def draw_critical_alert(self, frame, alert_payload):
        """
        Flashes a massive red warning on the screen if a fall is detected.
        """
        if not alert_payload:
            return frame
            
        h, w = frame.shape[:2]
        
        # Massive red border
        cv2.rectangle(frame, (0, 0), (w, h), self.colors["red"], 15)
        
        # Center warning text
        text = f"CRITICAL: {alert_payload['classification'].upper()}"
        text_size = cv2.getTextSize(text, self.font, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(frame, text, (text_x, h - 50), 
                    self.font, 1.2, self.colors["red"], 3)
                    
        return frame