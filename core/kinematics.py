import math
from collections import deque

class KinematicsEngine:
    def __init__(self, max_frames=30, fps=30.0):
        self.buffer = deque(maxlen=max_frames)
        self.last_known_state = None
        self.fps = fps

    def update(self, frame_data):
        if not frame_data or self._is_missing_core_joints(frame_data):
            if self.last_known_state is not None:
                self.buffer.append(self.last_known_state)
            return

        self.last_known_state = frame_data
        self.buffer.append(frame_data)

    def _is_missing_core_joints(self, frame_data):
        if frame_data.get("neck") is None:
            return True
        l_hip, r_hip = frame_data.get("hips", (None, None))
        if l_hip is None and r_hip is None:
            return True
        return False

    def is_ready(self):
        return len(self.buffer) >= 10  # Need at least a third of a second to get valid velocity

    def get_history(self):
        return list(self.buffer)

    def _get_mid_hip(self, frame_data):
        """Helper to safely calculate the center point between left and right hip."""
        l_hip, r_hip = frame_data.get("hips", (None, None))
        
        # If we have both, average them
        if l_hip and r_hip:
            return ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        # If one is occluded, just use the other
        elif l_hip:
            return l_hip
        elif r_hip:
            return r_hip
        return None

    def calculate_vertical_velocity(self):
        """
        Calculates vy in pixels per second.
        Positive velocity means moving DOWN (falling).
        """
        if not self.is_ready():
            return 0.0

        old_frame = self.buffer[0]
        new_frame = self.buffer[-1]

        old_hip = self._get_mid_hip(old_frame)
        new_hip = self._get_mid_hip(new_frame)

        if not old_hip or not new_hip:
            return 0.0

        dy = new_hip[1] - old_hip[1]
        
        # Time delta: (number of frames between old and new) / FPS
        dt = (len(self.buffer) - 1) / self.fps 

        # Velocity = pixels / second
        vy = dy / dt if dt > 0 else 0.0
        return vy

    def calculate_body_angle(self):
        """
        Calculates angle theta between mid-hip and neck relative to the horizontal axis.
        ~90 degrees = standing upright.
        ~0 or 180 degrees = lying flat on the ground.
        """
        if not self.is_ready():
            return 90.0 # Default to standing safe state

        current_frame = self.buffer[-1]
        neck = current_frame.get("neck")
        mid_hip = self._get_mid_hip(current_frame)

        if not neck or not mid_hip:
            return 90.0

        dx = neck[0] - mid_hip[0]
        # Image coordinates: y goes down. So we do hip_y - neck_y to keep standard geometry
        dy = mid_hip[1] - neck[1] 

        # atan2 handles all quadrants. We convert to degrees and take absolute 
        # so standing is always ~90 and lying down is ~0.
        angle = math.degrees(math.atan2(dy, dx))
        return abs(angle)