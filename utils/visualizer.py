import cv2
import numpy as np

class DebugVisualizer:
    """
    Renders real-time telemetry, diagnostic data, and alerts onto video frames.
    
    Provides visual feedback for:
    - Pose skeleton and detection confidence
    - Kinematic measurements (velocity, body angle)
    - FSM state and fall classification
    - Clinical diagnostics (gait metrics, agitation, immobility, wandering, seizure alerts)
    - Critical fall alerts with dynamic visual emphasis
    """
    
    def __init__(self):
        self.colors = {
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "black": (0, 0, 0),
            "orange": (0, 165, 255),
            "dark_gray": (30, 30, 30) # Background for the telemetry panel
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_yolo_skeleton(self, results):
        """
        Uses Ultralytics' built-in annotator to draw the bounding boxes and skeleton.
        Returns the annotated frame.
        """
        return results[0].plot()

    def draw_telemetry(self, frame, v_total, theta, current_state, buffer_size, classification="N/A", gait_metrics=None, immobility_data=None, agitation_data=None, wandering_data=None, seizure_data=None, safety_data=None, environmental_objects=None):
        """
        Render comprehensive telemetry and diagnostic data onto video frame.
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
        if seizure_data and "alerts" in seizure_data: 
            diagnostics.extend(seizure_data["alerts"]) 
        if safety_data and "alerts" in safety_data:
            diagnostics.extend(safety_data["alerts"])

        # 2. Draw Environmental Object Boxes (Sensor Fusion Visualizer)
        if environmental_objects:
            for obj in environmental_objects:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                # Orange bounding box for medical equipment
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["orange"], 2)
                cv2.putText(frame, obj['name'].upper(), (x1, y1 - 10), 
                            self.font, 0.5, self.colors["orange"], 2)

        # 3. Create the Dedicated Sidebar Panel
        h, w = frame.shape[:2]
        panel_width = 460 # Fixed width to prevent the window from resizing dynamically
        
        # Create a solid dark gray matrix to act as our HUD canvas
        sidebar = np.zeros((h, panel_width, 3), dtype=np.uint8)
        sidebar[:] = self.colors["dark_gray"]

        # 4. Draw all text onto the SIDEBAR instead of the video frame
        
        # Buffer Status
        buffer_color = self.colors["green"] if buffer_size >= 10 else self.colors["yellow"]
        cv2.putText(sidebar, f"Buffer: {buffer_size}/30", (15, 30), 
                    self.font, 0.6, buffer_color, 2)

        # Kinematics
        cv2.putText(sidebar, f"v_total: {v_total:.2f} px/s", (15, 60), 
                    self.font, 0.6, self.colors["white"], 2)
        cv2.putText(sidebar, f"Theta:   {theta:.1f} deg", (15, 90), 
                    self.font, 0.6, self.colors["white"], 2)

        # FSM / Posture State
        state_color = self.colors["red"] if current_state in ["RAPID_DESCENT", "NO_RECOVERY", "GROUND_CONTACT"] else self.colors["green"]
        cv2.putText(sidebar, f"STATE: {current_state}", (15, 120), 
                    self.font, 0.7, state_color, 2)

        # Fall Classification Type
        class_color = self.colors["yellow"] if classification != "N/A" else self.colors["white"]
        cv2.putText(sidebar, f"TYPE:  {classification}", (15, 150), 
                    self.font, 0.7, class_color, 2)

        # Gait Analyzer
        if gait_metrics:
            stride = gait_metrics.get("stride_length_px", 0.0)
            cadence = gait_metrics.get("cadence_spm", 0.0)
            cv2.putText(sidebar, f"Stride:  {stride} px", (15, 190), 
                        self.font, 0.6, self.colors["white"], 2)
            cv2.putText(sidebar, f"Cadence: {cadence} spm", (15, 220), 
                        self.font, 0.6, self.colors["white"], 2)

        # Immobility Tracker
        if immobility_data:
            timer = immobility_data.get("motionless_sec", 0.0)
            is_immobile = immobility_data.get("is_immobile", False)
            
            mins = int(timer // 60)
            secs = int(timer % 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            timer_color = self.colors["red"] if is_immobile else self.colors["white"]
            cv2.putText(sidebar, f"Motionless: {time_str}", (15, 250), 
                        self.font, 0.6, timer_color, 2)
        
        # Agitation Engine
        if agitation_data:
            score = agitation_data.get("score", 0)
            risk = agitation_data.get("risk", "low")
            
            score_color = self.colors["white"]
            if risk == "high": score_color = self.colors["red"]
            elif risk == "medium": score_color = self.colors["yellow"]
            
            cv2.putText(sidebar, f"Agitation: [{score}] Risk: {risk.upper()}", (15, 280), 
                        self.font, 0.6, score_color, 2)

        # Wandering / Spatial Zones
        if wandering_data:
            zone = wandering_data.get("current_zone", "Unknown")
            risk = wandering_data.get("risk", "Low")
            time_str = "Night" if wandering_data.get("is_night") else "Day"
            
            risk_color = self.colors["white"]
            if risk == "Critical": risk_color = self.colors["red"]
            elif risk == "High": risk_color = self.colors["yellow"]
            
            cv2.putText(sidebar, f"Zone: {zone} ({time_str})", (15, 310), 
                        self.font, 0.6, self.colors["white"], 2)
            cv2.putText(sidebar, f"Wandering Risk: {risk.upper()}", (15, 340), 
                        self.font, 0.6, risk_color, 2)

        # Seizure Module
        if seizure_data:
            event = seizure_data.get("event", "none")
            conf = seizure_data.get("confidence", 0.0)
            color = self.colors["red"] if conf >= 0.75 else self.colors["white"]
            
            cv2.putText(sidebar, f"Seizure: {event.upper()} [{conf:.2f}]", (15, 370), 
                        self.font, 0.6, color, 2)
        
        # Render prioritized alerts stacked sequentially
        y_offset = 420
        for alert in diagnostics:
            cv2.putText(sidebar, f"! {alert}", (15, y_offset), self.font, 0.6, self.colors["red"], 2)
            y_offset += 30

        # 5. Stitch the Sidebar and the Video Frame together horizontally
        # The sidebar goes on the left, the video goes on the right
        composite_frame = np.hstack((sidebar, frame))

        return composite_frame

    def draw_critical_alert(self, frame, alert_payload):
        """
        Render full-screen critical alert border and text.
        Because this is called AFTER draw_telemetry, it will wrap the entire composite window.
        """
        if not alert_payload:
            return frame
            
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), self.colors["red"], 15)
        
        text = f"CRITICAL: {alert_payload['classification'].upper()}"
        text_size = cv2.getTextSize(text, self.font, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(frame, text, (text_x, h - 50), 
                    self.font, 1.2, self.colors["red"], 3)
                    
        return frame