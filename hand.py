import cv2
import mediapipe as mp
import numpy as np
import time
import json
from collections import deque

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
p_time = 0

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

# FPS smoothing
fps_buffer = deque(maxlen=30)

# Coordinate smoothing buffers (reduces jitter for robotic control)
SMOOTHING_WINDOW = 5
coord_buffers = {
    'shoulder': deque(maxlen=SMOOTHING_WINDOW),
    'elbow': deque(maxlen=SMOOTHING_WINDOW),
    'wrist': deque(maxlen=SMOOTHING_WINDOW),
    'fingers': {name: deque(maxlen=SMOOTHING_WINDOW) for name in ['thumb', 'index', 'middle', 'ring', 'pinky']}
}

RECORD_DATA = False
recorded_data = []

# Custom drawing specification - only draw arm landmarks
ARM_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Define arm connections only
ARM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

FINGER_TIPS = {
    'thumb': mp_hands.HandLandmark.THUMB_TIP,
    'index': mp_hands.HandLandmark.INDEX_FINGER_TIP,
    'middle': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    'ring': mp_hands.HandLandmark.RING_FINGER_TIP,
    'pinky': mp_hands.HandLandmark.PINKY_TIP,
}


def smooth_coordinates(buffer, new_coord):
    """Apply moving average smoothing to reduce jitter"""
    buffer.append(new_coord)
    return np.mean(list(buffer), axis=0) if len(buffer) > 0 else new_coord

def extract_arm_coordinates(landmarks, handedness, w, h):
    """Extract and return arm coordinates for robotic control"""
    if handedness == "Left":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    else:
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    shoulder_coords = np.array([shoulder.x * w, shoulder.y * h, shoulder.z * h])
    elbow_coords = np.array([elbow.x * w, elbow.y * h, elbow.z * h])
    wrist_coords = np.array([wrist.x * w, wrist.y * h, wrist.z * h])
    
    # Apply smoothing
    shoulder_smooth = smooth_coordinates(coord_buffers['shoulder'], shoulder_coords)
    elbow_smooth = smooth_coordinates(coord_buffers['elbow'], elbow_coords)
    wrist_smooth = smooth_coordinates(coord_buffers['wrist'], wrist_coords)
    
    return {
        'shoulder': {'x': shoulder_smooth[0], 'y': shoulder_smooth[1], 'z': shoulder_smooth[2]},
        'elbow': {'x': elbow_smooth[0], 'y': elbow_smooth[1], 'z': elbow_smooth[2]},
        'wrist': {'x': wrist_smooth[0], 'y': wrist_smooth[1], 'z': wrist_smooth[2]}
    }

def extract_finger_coordinates(hand_landmarks, w, h):
    """Extract and return finger tip coordinates for robotic control"""
    finger_coords = {}
    
    for name, tip_idx in FINGER_TIPS.items():
        tip = hand_landmarks.landmark[tip_idx]
        coords = np.array([tip.x * w, tip.y * h, tip.z * h])
        
        # Apply smoothing
        smooth_coords = smooth_coordinates(coord_buffers['fingers'][name], coords)
        
        finger_coords[name] = {
            'x': smooth_coords[0],
            'y': smooth_coords[1],
            'z': smooth_coords[2]
        }
    
    return finger_coords

def draw_arm_landmarks(image, landmarks, w, h):
    """Draw arm landmarks and connections"""
    for start_idx, end_idx in ARM_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        start_pt = (int(start.x * w), int(start.y * h))
        end_pt = (int(end.x * w), int(end.y * h))
        
        cv2.line(image, start_pt, end_pt, (0, 255, 0), 3)
    
    for landmark_idx in ARM_LANDMARKS:
        lm = landmarks[landmark_idx]
        x, y = int(lm.x * w), int(lm.y * h)
        
        cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 10, (255, 255, 255), 2)

def draw_control_overlay(image, arm_data, finger_data, handedness, w, h):
    """Draw clean overlay showing current control data"""
    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 300), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # Header
    cv2.putText(image, f"ROBOT CONTROL - {handedness.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    y = 60
    # Arm coordinates
    cv2.putText(image, "ARM:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 20
    for joint, coords in arm_data.items():
        text = f"  {joint.capitalize()}: ({int(coords['x'])}, {int(coords['y'])}, {int(coords['z'])})"
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 18
    
    # Finger coordinates
    y += 5
    cv2.putText(image, "FINGERS:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 20
    for finger, coords in finger_data.items():
        text = f"  {finger.capitalize()}: ({int(coords['x'])}, {int(coords['y'])}, {int(coords['z'])})"
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 18

# Configure MediaPipe Pose and Hands
with mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose, \
     mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    p_time = time.time()
    print("=" * 50)
    print("PROSTHETIC ARM CONTROL SYSTEM")
    print("=" * 50)
    print("Controls:")
    print("  Q/ESC - Quit")
    print("  R - Toggle recording")
    print("  S - Save recorded data")
    print("=" * 50)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        # Flip image horizontally for selfie-view
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        
        # Get image dimensions
        h, w, c = image.shape
        
        # Draw and extract ARM landmarks only (shoulder, elbow, wrist)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            draw_arm_landmarks(image, landmarks, w, h)
            
        
        # Draw and extract HAND landmarks (for fingers)
        if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
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

                # Extract coordinates
                arm_data = extract_arm_coordinates(landmarks, handedness, w, h)
                finger_data = extract_finger_coordinates(hand_landmarks, w, h)
                
                # Display control data
                draw_control_overlay(image, arm_data, finger_data, handedness, w, h)
        
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
        
        
        key = cv2.waitKey(1) & 0xFF
        # Exit on 'q', 'esc' key
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('r') or key == ord('R'): # Toggle recording
            RECORD_DATA = not RECORD_DATA
            print(f"Recording: {'ON' if RECORD_DATA else 'OFF'}")
        # elif key == ord('s') or key == ord('S'): # Save recorded data
        #     if recorded_data:
        #         filename = f"robot_data_{int(time.time())}.json"
        #         with open(filename, 'w') as f:
        #             json.dump(recorded_data, f, indent=2)
        #         print(f"Saved {len(recorded_data)} frames to {filename}")
        #         recorded_data = []

cap.release()
cv2.destroyAllWindows()