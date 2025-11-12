import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
p_time = 0

# Initialize webcam
cap = cv2.VideoCapture(0)

# Custom drawing specification - only draw arm landmarks
arm_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Define arm connections only
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

# Configure MediaPipe Pose and Hands
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        # Flip image horizontally for selfie-view
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        
        # Get image dimensions
        h, w, c = image.shape
        
        # Draw and extract ARM landmarks only (shoulder, elbow, wrist)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Draw only arm landmarks and connections
            for connection in arm_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Convert to pixel coordinates
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                
                # Draw connection line
                cv2.line(image, start_point, end_point, (0, 255, 0), 3)
            
            # Draw arm landmarks as circles
            for landmark_idx in arm_landmarks:
                landmark = landmarks[landmark_idx]
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                z_px = landmark.z*h  # normalized depth, can also do lm.z * h for scaled value
                
                # Draw circle for each landmark
                cv2.circle(image, (x_px, y_px), 8, (0, 0, 255), -1)
                cv2.circle(image, (x_px, y_px), 10, (255, 255, 255), 2)
                
            
            # Extract coordinates
            
            
            
            
            
            
            

            # Display info on screen - LEFT ARM
            y_offset = 30
            
        
        # Draw and extract HAND landmarks (for fingers)
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Get handedness (left or right)
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label
            # Display info on screen - LEFT ARM
                y_offset = 30
                if handedness == "Left":
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    print("\n=== LEFT ARM ===")
                    print(f"Shoulder: Norm({left_shoulder.x:.3f}, {left_shoulder.y:.3f}, {left_shoulder.z:.3f}) | "
                        f"Px({int(left_shoulder.x*w)}, {int(left_shoulder.y*h)})")
                    print(f"ELBOW:    Norm({left_elbow.x:.3f}, {left_elbow.y:.3f}, {left_elbow.z:.3f}) | "
                        f"Px({int(left_elbow.x*w)}, {int(left_elbow.y*h)})")
                    print(f"Wrist:    Norm({left_wrist.x:.3f}, {left_wrist.y:.3f}, {left_wrist.z:.3f}) | "
                        f"Px({int(left_wrist.x*w)}, {int(left_wrist.y*h)})")
                    cv2.putText(image, "LEFT ARM", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Shoulder: ({int(left_shoulder.x*w)}, {int(left_shoulder.y*h)}, {int(left_shoulder.z*h)})", 
                            (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(image, f"ELBOW: ({int(left_elbow.x*w)}, {int(left_elbow.y*h)}, {int(left_elbow.z*h)})", 
                            (10, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(image, f"Wrist: ({int(left_wrist.x*w)}, {int(left_wrist.y*h)}, {int(left_wrist.z*h)})", 
                            (10, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else: 
                # Display info on screen - RIGHT ARM
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    print("\n=== RIGHT ARM ===")
                    print(f"Shoulder: Norm({right_shoulder.x:.3f}, {right_shoulder.y:.3f}, {right_shoulder.z:.3f}) | "
                        f"Px({int(right_shoulder.x*w)}, {int(right_shoulder.y*h)}, {int(right_shoulder.z*h)})")
                    print(f"ELBOW:    Norm({right_elbow.x:.3f}, {right_elbow.y:.3f}, {right_elbow.z:.3f}) | "
                        f"Px({int(right_elbow.x*w)}, {int(right_elbow.y*h)}, {int(right_elbow.z*h)})")
                    print(f"Wrist:    Norm({right_wrist.x:.3f}, {right_wrist.y:.3f}, {right_wrist.z:.3f}) | "
                        f"Px({int(right_wrist.x*w)}, {int(right_wrist.y*h)}, {int(right_wrist.z*h)})")
                    cv2.putText(image, "RIGHT ARM", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Shoulder: ({int(right_shoulder.x*w)}, {int(right_shoulder.y*h)}, {int(right_shoulder.z*h)})", 
                            (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(image, f"ELBOW: ({int(right_elbow.x*w)}, {int(right_elbow.y*h)}, {int(right_elbow.z*h)})", 
                            (10, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(image, f"Wrist: ({int(right_wrist.x*w)}, {int(right_wrist.y*h)}, {int(right_wrist.y*h)})", 
                            (10, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                print(f"\n=== {handedness} HAND ===")
                
                # Extract key finger points
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                print(f"Wrist:      Px({int(wrist.x*w)}, {int(wrist.y*h)}, {int(wrist.z*h)})")
                print(f"Thumb tip:  Px({int(thumb_tip.x*w)}, {int(thumb_tip.y*h)}, {int(thumb_tip.z*h)})")
                print(f"Index tip:  Px({int(index_tip.x*w)}, {int(index_tip.y*h)}, {int(index_tip.z*h)})")
                print(f"Middle tip: Px({int(middle_tip.x*w)}, {int(middle_tip.y*h)}, {int(middle_tip.z*h)})")
                print(f"Ring tip:   Px({int(ring_tip.x*w)}, {int(ring_tip.y*h)}, {int(ring_tip.z*h)})")
                print(f"Pinky tip:  Px({int(pinky_tip.x*w)}, {int(pinky_tip.y*h)}, {int(pinky_tip.z*h)})")

                
                # Display hand info on screen
                x_pos = w - 200
                y_pos = 30 + hand_idx * 150
                cv2.putText(image, f"{handedness} HAND", (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(image, f"Thumb:  ({int(thumb_tip.x*w)}, {int(thumb_tip.y*h)}, {int(thumb_tip.z*h)})", 
                        (x_pos, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(image, f"Index:  ({int(index_tip.x*w)}, {int(index_tip.y*h)}, {int(index_tip.z*h)})", 
                        (x_pos, y_pos+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(image, f"Middle: ({int(middle_tip.x*w)}, {int(middle_tip.y*h)}, {int(middle_tip.z*h)})", 
                        (x_pos, y_pos+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(image, f"Ring:   ({int(ring_tip.x*w)}, {int(ring_tip.y*h)}, {int(ring_tip.z*h)})", 
                        (x_pos, y_pos+105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(image, f"Pinky:  ({int(pinky_tip.x*w)}, {int(pinky_tip.y*h)}, {int(pinky_tip.z*h)})", 
                        (x_pos, y_pos+130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
        
        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        text = f'FPS: {int(fps)}'
        font_w, font_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        x_pos = image.shape[1] - font_w - 10   # width - text width - margin
        y_pos = image.shape[0] - 10         # height - margin
        # Display FPS on screen
        cv2.putText(image, text , (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Elbow + Hand Tracking', image)
        
        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()