import math

class SafetyMonitor:
    def __init__(self, requires_walker=False, on_oxygen=False, has_iv=False):
        # Patient Profile Flags
        self.requires_walker = requires_walker
        self.on_oxygen = on_oxygen
        self.has_iv = has_iv
        
        # Temporal tracking for sustained behaviors
        self.face_touch_frames = 0
        self.iv_touch_frames = 0
        
        # Thresholds
        self.max_touch_frames = 45 # 1.5 seconds of sustained pulling behavior

    def update(self, frame_data, posture, v_total):
        alerts = []
        risk_level = "Safe"
        
        head = frame_data.get("head")
        neck = frame_data.get("neck")
        l_wrist, r_wrist = frame_data.get("wrists", (None, None))
        l_elbow, r_elbow = frame_data.get("shoulders", (None, None)) # Using shoulders/elbow area for IV proxy

        # --- 1. OXYGEN / TUBE PULLING HEURISTIC ---
        if self.on_oxygen and head and neck:
            touching_face = False
            for wrist in [l_wrist, r_wrist]:
                if wrist:
                    # Check if wrist is extremely close to the nose/mouth/neck area
                    dist_to_face = math.hypot(wrist[0] - head[0], wrist[1] - head[1])
                    if dist_to_face < 40.0: # pixels
                        touching_face = True
            
            if touching_face:
                self.face_touch_frames += 1
            else:
                self.face_touch_frames = max(0, self.face_touch_frames - 2) # Cooldown

            if self.face_touch_frames > self.max_touch_frames:
                alerts.append("Oxygen/Tube Pulling Suspected")
                risk_level = "Critical"

        # --- 2. IV LINE PULLING HEURISTIC ---
        if self.has_iv and l_wrist and r_wrist:
            # Check if one hand is persistently grabbing the opposite arm (proxy for IV pull)
            dist_hands = math.hypot(l_wrist[0] - r_wrist[0], l_wrist[1] - r_wrist[1])
            if dist_hands < 30.0:
                self.iv_touch_frames += 1
            else:
                self.iv_touch_frames = max(0, self.iv_touch_frames - 2)
                
            if self.iv_touch_frames > self.max_touch_frames:
                alerts.append("IV Line Interference Suspected")
                risk_level = "High"

        # --- 3. UNASSISTED MOBILITY (WALKER) HEURISTIC ---
        if self.requires_walker and posture == "STANDING" and v_total > 15.0:
            # If they are moving, are their hands down by their sides, or extended forward holding a walker?
            hips = frame_data.get("hips", (None, None))
            if hips[0] and hips[1] and l_wrist and r_wrist:
                mid_hip_y = (hips[0][1] + hips[1][1]) / 2
                
                # If wrists are below the hips, arms are dangling. They aren't using a walker.
                if l_wrist[1] > mid_hip_y and r_wrist[1] > mid_hip_y:
                    alerts.append("Unassisted Mobility (Walker Required)")
                    risk_level = "High"

        return {
            "status": risk_level,
            "alerts": alerts
        }