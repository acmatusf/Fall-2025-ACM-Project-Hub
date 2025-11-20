import cv2
import mediapipe as mp

# Initialize MediaPipe Pose only (no hands)
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define arm landmarks - RIGHT ARM only
arm_landmarks = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Define arm connections
arm_connections = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

# Configure MediaPipe Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        # Get image dimensions
        h, w, c = image.shape
        
        # Convert BGR to RGB BEFORE flip (for detection)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image on ORIGINAL (non-flipped) image
        pose_results = pose.process(image_rgb)
        
        # NOW flip for display (selfie mode)
        image = cv2.flip(image, 1)
        
        # Calculate origin point: center-bottom of screen
        origin_x = w // 2
        origin_y = h  # Bottom of screen
        
        # Draw origin point (reference point)
        cv2.circle(image, (origin_x, origin_y - 10), 8, (0, 255, 255), -1)  # Cyan dot
        cv2.putText(image, "ORIGIN", (origin_x - 30, origin_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw and extract ARM landmarks
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Draw arm connections
            for connection in arm_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Convert to pixel coordinates
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                # FLIP X coordinates (because image is flipped)
                start_x = w - start_x
                end_x = w - end_x
                
                # Draw connection line
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
            
            # Draw arm landmarks as circles
            for landmark_idx in arm_landmarks:
                landmark = landmarks[landmark_idx]
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                
                # FLIP X coordinate
                x_px = w - x_px
                
                # Draw circles
                cv2.circle(image, (x_px, y_px), 8, (0, 0, 255), -1)
                cv2.circle(image, (x_px, y_px), 10, (255, 255, 255), 2)
            
            # Extract 3D coordinates
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Convert to pixel coordinates (X, Y)
            shoulder_x = int(right_shoulder.x * w)
            shoulder_y = int(right_shoulder.y * h)
            elbow_x = int(right_elbow.x * w)
            elbow_y = int(right_elbow.y * h)
            wrist_x = int(right_wrist.x * w)
            wrist_y = int(right_wrist.y * h)
            
            # FLIP X coordinates
            shoulder_x = w - shoulder_x
            elbow_x = w - elbow_x
            wrist_x = w - wrist_x
            
            # Get Z coordinates (depth) - in relative scale
            shoulder_z = right_shoulder.z
            elbow_z = right_elbow.z
            wrist_z = right_wrist.z
            
            # Calculate RELATIVE coordinates (from origin point)
            shoulder_rel_x = shoulder_x - origin_x
            shoulder_rel_y = shoulder_y - origin_y
            elbow_rel_x = elbow_x - origin_x
            elbow_rel_y = elbow_y - origin_y
            wrist_rel_x = wrist_x - origin_x
            wrist_rel_y = wrist_y - origin_y
            
            # Display info on screen - RIGHT ARM with 3D coordinates
            y_offset = 30
            cv2.putText(image, "RIGHT ARM - 3D Coordinates", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(image, f"Shoulder: ({shoulder_rel_x:4d}, {shoulder_rel_y:4d}, {shoulder_z:+.3f})", 
                       (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(image, f"ELBOW:    ({elbow_rel_x:4d}, {elbow_rel_y:4d}, {elbow_z:+.3f})", 
                       (10, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(image, f"Wrist:    ({wrist_rel_x:4d}, {wrist_rel_y:4d}, {wrist_z:+.3f})", 
                       (10, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Z-depth info
            cv2.putText(image, "Z: (-) = Closer to camera | (+) = Farther from camera", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display the image
        cv2.imshow('Right Arm Tracking', image)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

