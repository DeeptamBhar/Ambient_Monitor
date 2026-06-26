import cv2
import math
import numpy as np
from collections import defaultdict

class PatientIdentifier:
    def __init__(self, required_frames=30):
        self.required_frames = required_frames
        self.frame_count = 0
        
        # Dictionary to store the behavioral history of each track_id
        # { track_id: {"positions": [], "postures": [], "colors": []} }
        self.candidates = defaultdict(lambda: {"positions": [], "postures": [], "colors": []})
        self.locked_target_id = None

    def _extract_dominant_color(self, frame, bbox):
        """Extracts the mean HSV color from the center torso of the bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        # Grab a small crop of the center of the bounding box (the torso)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = frame[max(0, cy-20):cy+20, max(0, cx-20):cx+20]
        
        if crop.size == 0:
            return (0, 0, 0) # Fallback if out of bounds
            
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_crop)[:3]
        return mean_hsv

    def _calculate_posture(self, keypoints):
        """A lightweight posture check specifically for initialization."""
        head = keypoints[0]
        hips = keypoints[11] # Left hip
        
        # If head and hips are roughly on the same Y-plane, they are lying down
        if abs(head[1] - hips[1]) < 50:
            return "LYING DOWN"
        return "STANDING"

    def update(self, frame, track_ids, bboxes, all_keypoints, frame_width):
        """Feeds the matrix for 1 second, then returns the locked ID."""
        if self.locked_target_id is not None:
            return self.locked_target_id

        self.frame_count += 1

        # Phase 1: Data Collection (The 1-Second Observation Window)
        for idx, t_id in enumerate(track_ids):
            bbox = bboxes[idx]
            keypoints = all_keypoints[idx]
            
            # 1. Location (Center point)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self.candidates[t_id]["positions"].append((cx, cy))
            
            # 2. Posture
            posture = self._calculate_posture(keypoints)
            self.candidates[t_id]["postures"].append(posture)
            
            # 3. Color
            color = self._extract_dominant_color(frame, bbox)
            self.candidates[t_id]["colors"].append(color)

        # Phase 2: The Multi-Factor Matrix Evaluation
        if self.frame_count >= self.required_frames:
            best_score = -999
            winning_id = None
            
            for t_id, data in self.candidates.items():
                score = 0
                
                # --- METRIC A: KINEMATIC PROFILING (35 pts) ---
                # A nurse walks across the room (high distance). A patient stays put.
                start_pos = data["positions"][0]
                end_pos = data["positions"][-1]
                distance_traveled = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                
                if distance_traveled < 50: score += 35      # Stationary resident
                elif distance_traveled > 150: score -= 20   # Fast moving transient
                
                # --- METRIC B: POSTURE MATRIX (35 pts) ---
                # If they spent the majority of the 1-second window lying down, huge bonus
                lying_down_frames = data["postures"].count("LYING DOWN")
                if lying_down_frames > (self.required_frames * 0.5):
                    score += 35
                
                # --- METRIC C: EDGE REJECTION (15 pts) ---
                # Where did they end up at the end of the 1 second?
                last_x = data["positions"][-1][0]
                margin = frame_width * 0.15 # 15% edge boundary
                if margin < last_x < (frame_width - margin):
                    score += 15 # They are in the center of the room
                else:
                    score -= 15 # They are touching the door/edge
                    
                # --- METRIC D: COLOR HISTOGRAM (15 pts) ---
                # Assuming hospital gowns are light (high value/low saturation) and scrubs are dark
                mean_saturation = np.mean([c[1] for c in data["colors"]])
                if mean_saturation < 100: # Light/pale colors (Gown)
                    score += 15
                else:                     # Highly saturated colors (Scrubs)
                    score -= 10
                    
                print(f"Candidate ID {t_id} | Score: {score}")
                
                if score > best_score:
                    best_score = score
                    winning_id = t_id
                    
            self.locked_target_id = winning_id
            print(f" MATRIX LOCK-ON: Assigned Patient Track ID: {self.locked_target_id}")
            
        return self.locked_target_id