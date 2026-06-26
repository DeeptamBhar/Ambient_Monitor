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

    def _check_intersection(self, box1, box2):
        """
        Checks if two bounding boxes [x1, y1, x2, y2] intersect in 2D space.
        Used to determine if the patient is interacting with a medical object.
        """
        if not box1 or not box2: 
            return False
        
        # If one rectangle is on the left side of the other
        if box1[0] > box2[2] or box2[0] > box1[2]: 
            return False
        # If one rectangle is above the other
        if box1[3] < box2[1] or box2[3] < box1[1]: 
            return False
        return True

    def update(self, frame_data, posture, v_total, environmental_objects=None, patient_bbox=None):
        alerts = []
        risk_level = "Safe"
        
        # Safely default to empty list if no objects passed
        if environmental_objects is None:
            environmental_objects = []

        head = frame_data.get("head")
        neck = frame_data.get("neck")
        l_wrist, r_wrist = frame_data.get("wrists", (None, None))

        # --- 1. SENSOR FUSION: MEDICAL SUPPORT INTERACTION ---
        # Check if the patient's bounding box overlaps with a walker or chair
        using_support = False
        for obj in environmental_objects:
            obj_bbox = obj['bbox']
            
            # If the patient bbox touches the equipment bbox, we assume they have physical support
            if self._check_intersection(patient_bbox, obj_bbox):
                using_support = True

        # --- 2. OXYGEN / TUBE PULLING HEURISTIC ---
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

        # --- 3. IV LINE PULLING HEURISTIC ---
        if self.has_iv and l_wrist and r_wrist:
            # Check if one hand is persistently grabbing the opposite arm (proxy for IV pull)
            dist_hands = math.hypot(l_wrist[0] - r_wrist[0], l_wrist[1] - r_wrist[1])
            if dist_hands < 30.0:
                self.iv_touch_frames += 1
            else:
                self.iv_touch_frames = max(0, self.iv_touch_frames - 2)
                
            if self.iv_touch_frames > self.max_touch_frames:
                alerts.append("IV Line Interference Suspected")
                # Don't downgrade a critical oxygen alert to high
                if risk_level != "Critical":
                    risk_level = "High"

        # --- 4. UNASSISTED MOBILITY HEURISTIC (Context-Aware) ---
        if self.requires_walker and posture == "STANDING" and v_total > 15.0:
            # If they are moving and NOT intersecting with the walker bounding box, flag it!
            if not using_support:
                hips = frame_data.get("hips", (None, None))
                if hips[0] and hips[1] and l_wrist and r_wrist:
                    mid_hip_y = (hips[0][1] + hips[1][1]) / 2
                    
                    # If wrists are below the hips, arms are dangling (not reaching for support)
                    if l_wrist[1] > mid_hip_y and r_wrist[1] > mid_hip_y:
                        alerts.append("Unassisted Mobility (Walker Abandoned!)")
                        if risk_level != "Critical":
                            risk_level = "High"

        return {
            "status": risk_level,
            "alerts": alerts,
            "using_support": using_support
        }