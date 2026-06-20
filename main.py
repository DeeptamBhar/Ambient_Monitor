import cv2
import yaml
import argparse
from ultralytics import YOLO
from core import FallDetectorFSM, KinematicsEngine, GaitAnalyzer, ImmobilityTracker, AgitationDetector, WanderingDetector
from utils import DebugVisualizer

def main():
    """
    Main entry point for the Ambient Intelligence fall detection pipeline.
    
    This function orchestrates the complete computer vision and clinical analysis pipeline:
    1. Loads YOLO model and video source (webcam or file)
    2. Extracts human pose keypoints from each frame using YOLOv8
    3. Processes kinematics data through the Finite State Machine for fall detection
    4. Analyzes clinical indicators: gait patterns, immobility, agitation, and wandering behavior
    5. Renders real-time telemetry and alerts to the video output
    
    Configuration is loaded from config.yaml and includes parameters for all detection modules.
    Press 'q' during execution to quit.
    
    Returns:
        None. Runs until video ends or user quits.
    """
    # Set up argparse 
    parser = argparse.ArgumentParser(description="Run the Fall Detection Pipeline")
    parser.add_argument('--source', type=str, default='0', help="Path to video file or '0' for webcam")
    args = parser.parse_args()

    # LOAD CONFIGURATION
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize your modules using the config file
    model = YOLO(config['model']['weights'])
    
    if args.source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)

    # Initialize the Rolling Buffer
    kinematics = KinematicsEngine(
        max_frames=config['kinematics']['max_frames'],
        fps=config['kinematics']['fps']
    )

    # Initialize the Brain (using ** unpacks the dictionary directly into the class kwargs)
    fsm = FallDetectorFSM(**config['fsm'])

    hud = DebugVisualizer()

    # Initialize the Gait Analyzer
    gait = GaitAnalyzer(fps=config['kinematics']['fps'])

    # Initialize Immobility Tracker
    immobility = ImmobilityTracker(**config['immobility'])
    agitation = AgitationDetector(**config['agitation'])
    wandering = WanderingDetector(**config['wandering'])

    print("Pipeline initialized. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or failed to read frame.")
            break
        results = model(frame, verbose=False)
        frame_data = {}
        frame_height, frame_width = frame.shape[:2] #Grab dimensions

        # PERCEPTION EXTRACTION 
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            confidences = results[0].keypoints.conf[0].cpu().numpy()

            def get_pt(idx):
                # Pulling the confidence threshold from the YAML file
                if confidences[idx] > config['model']['confidence_threshold']:
                    return (int(keypoints[idx][0]), int(keypoints[idx][1]))
                return None

            # Extract specific keypoints
            head = get_pt(0) 
            l_shoulder, r_shoulder = get_pt(5), get_pt(6)
            l_wrist, r_wrist = get_pt(9), get_pt(10) # NEW: Grab the wrists
            l_hip, r_hip = get_pt(11), get_pt(12)
            l_knee, r_knee = get_pt(13), get_pt(14)
            l_ankle, r_ankle = get_pt(15), get_pt(16)

            # Calculate Neck (Midpoint of shoulders)
            neck = None
            if l_shoulder and r_shoulder:
                neck = (int((l_shoulder[0] + r_shoulder[0]) / 2), 
                        int((l_shoulder[1] + r_shoulder[1]) / 2))

            # Pack the dictionary
            frame_data = {
                "head": head,
                "neck": neck,
                "shoulders": (l_shoulder, r_shoulder),
                "wrists": (l_wrist, r_wrist), # NEW: Pass wrists to the engine
                "hips": (l_hip, r_hip),
                "knees": (l_knee, r_knee),
                "ankles": (l_ankle, r_ankle)
            }

        # TEMPORAL MEMORY 
        kinematics.update(frame_data)

        # Calculate velocity first so we can feed it to the gait analyzer
        v_total = kinematics.calculate_velocity_vector()[2] if kinematics.is_ready() else 0.0

        # Gait engine builds its own peak-detection buffer independent of kinematics
        gait_metrics = gait.update(frame_data, v_total)

        # Start with safe default states if buffer isn't ready
        current_state = "STANDING"
        
        if kinematics.is_ready():
            
            vx, vy, v_total = kinematics.calculate_velocity_vector()
            theta = kinematics.calculate_body_angle()
            
            # Feed the FSM the total magnitude AND the vertical drop
            current_state = fsm.update(v_total, vy, theta)
            
            # THE DEBUG VISUALIZER 
            frame = hud.draw_yolo_skeleton(results)
            buffer_size = len(kinematics.get_history())

            # CLINICAL LOGIC
            # --- STEP 4.5: CLINICAL LOGIC ---
            immobility_data = immobility.update(v_total, theta)
            agitation_data = agitation.update(frame_data, theta)
            wandering_data = wandering.update(frame_data, frame_width, frame_height) 
            
            # Live Classification Logic for the HUD
            live_class = "N/A"
            if current_state in ["GROUND_CONTACT", "NO_RECOVERY"]:
                # Grab the payload to read the classification string
                temp_payload = fsm.generate_alert_payload(frame_data)
                if temp_payload:
                    live_class = temp_payload["classification"]
            # Pass all module data into the HUD
            frame = hud.draw_telemetry(
                frame, 
                v_total, 
                theta, 
                current_state, 
                buffer_size, 
                classification=live_class, 
                gait_metrics=gait_metrics,
                immobility_data=immobility_data,
                agitation_data=agitation_data,
                wandering_data=wandering_data
            )
        
            # Trigger full screen alert UI if timer runs out
            if current_state == "NO_RECOVERY":
                alert = fsm.generate_alert_payload(frame_data)
                frame = hud.draw_critical_alert(frame, alert)

        # Render the final output
        cv2.imshow("Ambient Intelligence - Pipeline", frame)
        
        # Press 'q' or 'Q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
