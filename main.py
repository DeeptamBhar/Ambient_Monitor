import cv2
import yaml
import argparse
from ultralytics import YOLO
from datetime import datetime
from core import FallDetectorFSM, KinematicsEngine, GaitAnalyzer, ImmobilityTracker, AgitationDetector, WanderingDetector, SeizureDetector, SafetyMonitor
from core.identification import PatientIdentifier
from utils import DebugVisualizer

def main():
    """
    Main entry point for the fall detection pipeline.
    
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
    object_model = YOLO("hospital_weights.pt")
    
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
    seizure = SeizureDetector(**config['seizure'])

    # Initialize the Safety Monitor (Configure based on your patient needs)
    safety = SafetyMonitor(requires_walker=True, on_oxygen=False, has_iv=False)
    identifier = PatientIdentifier(required_frames=30)

    print("Pipeline initialized. Press 'q' to quit.")

    active_alerts_memory = set()
    target_track_id = None
    detected_commodities = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or failed to read frame.")
            break
        
        # Run pose tracking with persistence enabled
        results = model.track(frame, persist=True, verbose=False)
        # Run custom object detection for hospital commodities
        obj_results = object_model(frame, verbose=False)

        frame_data = {}
        frame_height, frame_width = frame.shape[:2] #Grab dimensions

        # --- EXTRACT ENVIRONMENTAL OBJECTS (Walker = Class 0) ---
        environmental_objects = []
        if obj_results[0].boxes is not None:
            obj_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
            obj_classes = obj_results[0].boxes.cls.cpu().numpy()
            obj_confidences = obj_results[0].boxes.conf.cpu().numpy()
            
            for i in range(len(obj_boxes)):
                if obj_confidences[i] > 0.5:
                    class_id = int(obj_classes[i])
                    if class_id == 0: 
                        name = "walker"
                        environmental_objects.append({"name": name, "bbox": obj_boxes[i].tolist()})
        
                    # 2. ONLY LOG IF NOT ALREADY IN THE SET
                        if name not in detected_commodities:
                            print(f"\033[94m[COMMODITY DETECTED] {name.upper()} now in view.\033[0m")
                            detected_commodities.add(name)

        patient_bbox = None

        # --- EXTRACT POSE KEYPOINTS & PATIENT BOUNDING BOX ---
        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            all_keypoints = results[0].keypoints.xy.cpu().numpy()
            all_confidences = results[0].keypoints.conf.cpu().numpy()
            all_bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Pass everything to the Matrix. It will return None for the first 30 frames.
            target_track_id = identifier.update(frame, track_ids, all_bboxes, all_keypoints, frame_width)

            if target_track_id is None:
                # Still initializing. Show a "CALIBRATING" message on the HUD and skip processing this frame.
                cv2.putText(frame, "SYSTEM INITIALIZING (Scoring Matrix Active)...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Ambient Intelligence - Pipeline", frame)
                cv2.waitKey(1)
                continue # Skip the rest of the loop until we have a target

            # Once we have a lock, extract their specific keypoints
            if target_track_id in track_ids:
                target_idx = track_ids.index(target_track_id)
                keypoints = all_keypoints[target_idx]
                confidences = all_confidences[target_idx]
                patient_bbox = all_bboxes[target_idx].tolist()
            else:
                patient_bbox = None

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

            # NEW: Calculate actual physical posture based on thighs
            posture = kinematics.calculate_posture()
            
            # Feed the FSM the total magnitude AND the vertical drop
            current_state = fsm.update(v_total, vy, theta)

            # If the FSM says they are "STANDING" (safe), override it with the actual posture
            display_state = posture if current_state == "STANDING" else current_state
            
            # THE DEBUG VISUALIZER 
            frame = hud.draw_yolo_skeleton(results)
            buffer_size = len(kinematics.get_history())

            # CLINICAL LOGIC
            # --- STEP 4.5: CLINICAL LOGIC ---
            immobility_data = immobility.update(v_total, theta)
            agitation_data = agitation.update(frame_data, theta)
            wandering_data = wandering.update(frame_data, frame_width, frame_height) 
            
            # FUSE PIPELINE DATA FOR SEIZURE CHECK
            seizure_data = seizure.update(frame_data, current_state, immobility_data)

            # --- SENSOR FUSION SAFETY EVALUATION ---
            safety_data = safety.update(
                frame_data, 
                display_state, 
                v_total, 
                environmental_objects=environmental_objects, 
                patient_bbox=patient_bbox
            )

            # --- STEP 4.6: TERMINAL LOGGING ---
            # 1. Harvest all active alerts from the modules
            current_frame_alerts = set()
            if gait_metrics and "diagnostics" in gait_metrics:
                current_frame_alerts.update(gait_metrics["diagnostics"])
            if immobility_data and "alerts" in immobility_data:
                current_frame_alerts.update(immobility_data["alerts"])
            if agitation_data and "alerts" in agitation_data:
                current_frame_alerts.update(agitation_data["alerts"])
            if wandering_data and "alerts" in wandering_data:
                current_frame_alerts.update(wandering_data["alerts"])
            if seizure_data and "alerts" in seizure_data:             
                current_frame_alerts.update(seizure_data["alerts"])   
            if safety_data and "alerts" in safety_data:
                current_frame_alerts.update(safety_data["alerts"])

            # 2. Find alerts that just triggered on this exact frame
            new_alerts = current_frame_alerts - active_alerts_memory
            
            # 3. Print them in red with a timestamp
            for alert in new_alerts:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # \033[91m makes it bright red, \033[0m resets the color
                print(f"\033[0m[{timestamp}] SYSTEM ALERT: {alert}\033[0m")
                
            # 4. Update the memory for the next frame
            active_alerts_memory = current_frame_alerts 
            
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
                display_state, 
                buffer_size, 
                classification=live_class, 
                gait_metrics=gait_metrics,
                immobility_data=immobility_data,
                agitation_data=agitation_data,
                wandering_data=wandering_data,
                seizure_data=seizure_data,
                safety_data=safety_data,
                environmental_objects=environmental_objects
            )
        
            # Trigger full screen alert UI if timer runs out
            if current_state == "NO_RECOVERY":
                alert = fsm.generate_alert_payload(frame_data)
                frame = hud.draw_critical_alert(frame, alert)

        # If no commodities are detected in this frame, clear the set
        # so that the system is ready to log them again next time they appear.
        if len(environmental_objects) == 0:
            detected_commodities.clear()

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
