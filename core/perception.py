import cv2
import argparse
from ultralytics import YOLO

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run YOLOv8 Pose on a video or webcam.")
parser.add_argument('--source', type=str, default='0', help="Path to video file or '0' for webcam")
args = parser.parse_args()

model = YOLO('yolov8n-pose.pt')

# Handle the input type (convert '0' back to an integer if it's the webcam)
if args.source == '0':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.source)

# COCO Keypoint Dictionary for YOLOv8
# 0: Nose, 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip
# 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame or end of video.")
        break

    # Run YOLOv8 inference on the frame
    # verbose=False stops it from spamming your terminal every frame
    results = model(frame, verbose=False)
    
    # We need a place to store coordinates for this specific frame
    frame_data = {}

    # Check if a person is detected
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        
        # Grab the keypoints for the first person detected [Person 0]
        # .cpu().numpy() moves the tensor data to a standard python array
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidences = results[0].keypoints.conf[0].cpu().numpy()

        # Only extract the point if the confidence is decent (> 0.5)
        def get_pt(idx):
            if confidences[idx] > 0.5:
                return (int(keypoints[idx][0]), int(keypoints[idx][1]))
            return None

        # Extract the specific keypoints you requested
        head = get_pt(0) # Using Nose to approximate Head
        l_shoulder, r_shoulder = get_pt(5), get_pt(6)
        l_hip, r_hip = get_pt(11), get_pt(12)
        l_knee, r_knee = get_pt(13), get_pt(14)
        l_ankle, r_ankle = get_pt(15), get_pt(16)

        # Calculate the Neck (Midpoint of shoulders)
        neck = None
        if l_shoulder and r_shoulder:
            neck = (int((l_shoulder[0] + r_shoulder[0]) / 2), 
                    int((l_shoulder[1] + r_shoulder[1]) / 2))

        # Pack it up for the next stage of your pipeline
        frame_data = {
            "head": head,
            "neck": neck,
            "shoulders": (l_shoulder, r_shoulder),
            "hips": (l_hip, r_hip),
            "knees": (l_knee, r_knee),
            "ankles": (l_ankle, r_ankle)
        }

        print(f"Extracted Frame Data: {frame_data}")

        # --- DEBUG VISUALIZER ---
        # Draw the neck point just to prove the calculation works
        if neck:
            cv2.circle(frame, neck, 5, (0, 0, 255), -1) 
            cv2.putText(frame, 'Neck', (neck[0] + 10, neck[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # YOLO has a built-in annotator to draw the rest of the skeleton automatically
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("Ambient Intelligence - Perception Layer", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()