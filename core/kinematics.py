import math
from collections import deque

class KinematicsEngine:
    """
    Processes temporal joint position data to calculate human body kinematics.
    
    Maintains a rolling buffer of pose frames and computes velocity vectors and body angle.
    Requires minimum 10 frames of valid data before providing reliable kinematic estimates.
    """
    
    def __init__(self, max_frames=30, fps=30.0):
        """
        Initialize the KinematicsEngine with a circular buffer for temporal history.
        
        Args:
            max_frames (int, optional): Maximum number of frames to retain in the buffer. Defaults to 30.
                This determines the time window for velocity calculations: with fps=30, 30 frames ≈ 1 second.
            fps (float, optional): Frames per second of the input video stream. Defaults to 30.0.
                Used to convert pixel distances to velocity units (pixels/second).
        """
        self.buffer = deque(maxlen=max_frames)
        self.last_known_state = None
        self.fps = fps

    def update(self, frame_data):
        """
        Add a new frame of pose data to the temporal history buffer.
        
        If frame_data is invalid or missing critical joints (neck or hips), the last known valid state
        is replicated to maintain continuity in the buffer.
        
        Args:
            frame_data (dict): Dictionary containing joint positions. Expected keys:
                'neck', 'shoulders', 'hips', 'knees', 'ankles', 'head', 'wrists'.
                Joint values are tuples (x, y) in pixel coordinates, or None if not detected.
        
        Returns:
            None. Updates internal buffer state.
        """
        if not frame_data or self._is_missing_core_joints(frame_data):
            if self.last_known_state is not None:
                self.buffer.append(self.last_known_state)
            return

        self.last_known_state = frame_data
        self.buffer.append(frame_data)

    def _is_missing_core_joints(self, frame_data):
        """
        Check if critical joints required for kinematic calculations are present.
        
        Critical joints for fall detection are: neck (computed from shoulders) and hips.
        A valid frame requires both the neck position and at least one hip position.
        
        Args:
            frame_data (dict): Frame containing joint positions as (x, y) tuples or None.
        
        Returns:
            bool: True if frame is missing critical joints, False if all required joints present.
        """
        if frame_data.get("neck") is None:
            return True
        l_hip, r_hip = frame_data.get("hips", (None, None))
        if l_hip is None and r_hip is None:
            return True
        return False

    def is_ready(self):
        """
        Check if the buffer has enough temporal data for reliable kinematic calculations.
        
        Returns:
            bool: True if buffer contains at least 10 frames (typically ~0.33 seconds at 30 fps),
                False otherwise. Minimum 10 frames ensures velocity calculations aren't noisy.
        """
        return len(self.buffer) >= 10  # Need at least a third of a second to get valid velocity

    def get_history(self):
        """
        Retrieve the complete temporal history of pose frames.
        
        Returns:
            list: List of frame_data dictionaries, oldest to newest. Returns empty list if buffer is empty.
        """
        return list(self.buffer)

    def _get_mid_hip(self, frame_data):
        """
        Calculate the center point between left and right hip joints.
        
        Provides a robust hip position even if one hip is occluded. If both hips are visible,
        returns their midpoint. If one is occluded, returns the visible one. Returns None if
        neither hip is detected.
        
        Args:
            frame_data (dict): Frame data containing 'hips' key with tuple (left_hip, right_hip)
                where each is (x, y) in pixels or None.
        
        Returns:
            tuple or None: (x, y) midpoint in pixel coordinates, or None if no hips detected.
        """
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

    def calculate_velocity_vector(self):
        """
        Calculates vx, vy, and total velocity magnitude (v_total) in pixels per second.
        """
        if not self.is_ready():
            return 0.0, 0.0, 0.0

        old_hip = self._get_mid_hip(self.buffer[0])
        new_hip = self._get_mid_hip(self.buffer[-1])

        if not old_hip or not new_hip:
            return 0.0, 0.0, 0.0

        # Calculate distance traveled
        dx = new_hip[0] - old_hip[0]
        dy = new_hip[1] - old_hip[1]
        dt = (len(self.buffer) - 1) / self.fps 

        # Calculate vectors
        vx = abs(dx / dt) if dt > 0 else 0.0
        vy = dy / dt if dt > 0 else 0.0
        v_total = math.sqrt(vx**2 + vy**2) # Full magnitude of the crash
        
        return vx, vy, v_total

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