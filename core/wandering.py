import time
from datetime import datetime
from collections import deque

class WanderingDetector:
    """
    Detects unsafe wandering behavior, particularly nighttime exits and unsafe location access.
    
    Monitors patient location relative to defined zones and temporal context (day vs night).
    Flags critical alerts for night-time room exits, repeated door checking, and access to
    unsafe areas. Provides contextual risk scoring based on behavioral patterns.
    """
    
    def __init__(self, zones, night_start_hr=22, night_end_hr=6, door_check_window=300):
        """
        Initialize the wandering detector with spatial and temporal parameters.
        
        Args:
            zones (dict): Dictionary mapping zone names to bounding boxes.
                Format: {"zone_name": [x1, y1, x2, y2]}
                Coordinates should be normalized 0.0-1.0 (relative to frame dimensions).
                Example: {"Door": [0.8, 0.4, 1.0, 0.6], "Bed": [0.0, 0.0, 0.3, 0.3]}
            night_start_hr (int, optional): Hour (0-23) when night period begins. Defaults to 22 (10 PM).
            night_end_hr (int, optional): Hour (0-23) when night period ends. Defaults to 6 (6 AM).
            door_check_window (int, optional): Time window (seconds) to count door approach frequency.
                Defaults to 300 (5 minutes).
        """
        # Dictionary of zones: {"name": [x1, y1, x2, y2]} in normalized coordinates (0.0 to 1.0)
        self.zones = zones 
        
        # Temporal logic
        self.night_start_hr = night_start_hr # 22 = 10:00 PM
        self.night_end_hr = night_end_hr     # 6 = 6:00 AM
        
        # Track how many times they approach the door in X seconds (default 5 mins)
        self.door_check_window = door_check_window
        self.door_visits = deque()
        
        self.current_zone = "Unknown"

    def update(self, frame_data, frame_width, frame_height):
        """
        Update wandering detection based on current patient location and time of day.
        
        Performs spatial zone detection and temporal night-time checking. Tracks repeated
        door approaches and flags alerts for unsafe behaviors.
        
        Args:
            frame_data (dict): Frame containing joint positions, specifically 'hips' for location.
            frame_width (int): Width of the video frame in pixels.
            frame_height (int): Height of the video frame in pixels.
        
        Returns:
            dict: Risk assessment containing:
                'risk' (str): Risk level - 'Low', 'Medium', 'High', or 'Critical'
                'current_zone' (str): Current detected zone or 'Floor'/'Exited'
                'is_night' (bool): True if current time is within night hours
                'alerts' (list): List of specific alert messages
        """
        # 1. Temporal Check: Is it night time?
        current_hour = datetime.now().hour
        is_night = (current_hour >= self.night_start_hr) or (current_hour < self.night_end_hr)
        
        # 2. Spatial Check: Where are they?
        hips = frame_data.get("hips", (None, None))
        new_zone = "Floor" # Default generic zone
        
        if hips[0] and hips[1]:
            # Calculate Center of Mass
            mid_x = (hips[0][0] + hips[1][0]) / 2
            mid_y = (hips[0][1] + hips[1][1]) / 2
            
            # Normalize coordinates to 0.0 - 1.0
            norm_x = mid_x / frame_width
            norm_y = mid_y / frame_height
            
            # Check intersection with defined zones
            for zone_name, box in self.zones.items():
                x1, y1, x2, y2 = box
                if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                    new_zone = zone_name
                    break
        else:
            # If YOLO loses the hips completely, they might have left the frame.
            # If they were just at the door, we flag an exit.
            if self.current_zone == "Door":
                new_zone = "Exited"
        
        # 3. Track Repetitive Behavior
        current_time = time.time()
        # If they just entered the door zone
        if new_zone == "Door" and self.current_zone != "Door":
            self.door_visits.append(current_time)
            
        self.current_zone = new_zone
        
        # Prune old door visits outside our time window
        while self.door_visits and current_time - self.door_visits[0] > self.door_check_window:
            self.door_visits.popleft()
            
        return self._generate_score(is_night)

    def _generate_score(self, is_night):
        """
        Map current behavior to clinical risk level with specific alerts.
        
        Risk assessment rules:
        - Critical: Unsafe location, night-time exit, or immediate danger
        - High: Repeated door checking during night (potential exit planning)
        - Medium: Out of bed during night, or repeated door checking during day
        - Low: Normal daytime activity
        
        Args:
            is_night (bool): True if current time is within configured night hours.
        
        Returns:
            dict: Containing 'risk' (str), 'current_zone' (str), 'is_night' (bool), 'alerts' (list)
        """
        risk = "Low"
        alerts = []
        
        if self.current_zone == "Unsafe":
            risk = "Critical"
            alerts.append("Unsafe Location Intrusion!")
            
        elif self.current_zone == "Exited" and is_night:
            risk = "Critical"
            alerts.append("Night-time Room Exit Detected!")
            
        elif len(self.door_visits) >= 3:
            risk = "High" if is_night else "Medium"
            alerts.append("Repeated Door Checking")
            
        elif self.current_zone == "Floor" and is_night:
            # Out of bed at night is a baseline medium risk for dementia patients
            if risk == "Low": risk = "Medium" 
            
        return {
            "risk": risk,
            "current_zone": self.current_zone,
            "is_night": is_night,
            "alerts": alerts
        }
    
