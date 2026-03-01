# -------------------------
# IMPORTS
# -------------------------
import ikpy.chain
import numpy as np
import math
import time
import cv2
import mediapipe as mp


# -------------------------
# MediaPipe Setup
# -------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)


# -------------------------
# Depth Calibration
# -------------------------
DEPTH_MIN = None
DEPTH_MAX = None

start_time = time.time()

CALIB_FAR_TIME = 16
CALIB_CLOSE_TIME = 16


# -------------------------
# IK Setup
# -------------------------
my_chain = ikpy.chain.Chain.from_urdf_file(
    "Hardware/actual_arm_urdf.urdf",
    active_links_mask=[False, True, True, True, True, True]
)

target_position = [0, 0.2, 0.1]
target_orientation = [-1, 0, 0]

ik = my_chain.inverse_kinematics(
    target_position,
    target_orientation,
    orientation_mode="Y"
)


# -------------------------
# IK Wrapper (NO ARDUINO)
# -------------------------
def doIK():
    global ik
    old_position = ik.copy()

    ik = my_chain.inverse_kinematics(
        target_position,
        target_orientation,
        orientation_mode=None,
        initial_position=old_position
    )

    joint_degrees = list(map(lambda r: round(math.degrees(r), 2), ik.tolist()))
    print("\nJoint Angles (degrees):", joint_degrees)


def move(x, y, z):
    global target_position
    target_position = [x, y, z]
    doIK()
    print(f"Scaled Target -> X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")


# -------------------------
# Camera Loop
# -------------------------
frame_count = 0
PLOT_EVERY_N = 200

print("Starting camera. Calibrate first.")

while cap.isOpened():

    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    elapsed = time.time() - start_time

    # -------------------------
    # Calibration Status
    # -------------------------
    if elapsed < CALIB_FAR_TIME:
        status = "MOVE HAND FAR"
    elif elapsed < CALIB_FAR_TIME + CALIB_CLOSE_TIME:
        status = "MOVE HAND CLOSE"
    else:
        status = "CALIBRATION DONE"

    neck_ref = None

    # -------------------------
    # Neck Reference
    # -------------------------
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        neck_ref = {
            "x": (ls.x + rs.x) / 2,
            "shoulder_y": (ls.y + rs.y) / 2
        }

    # -------------------------
    # Hand Tracking
    # -------------------------
    if hand_results.multi_hand_landmarks and neck_ref:

        hand = hand_results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        wrist = lm[0]
        index = lm[8]
        thumb = lm[4]
        pinky = lm[20]

        # Claw logic (display only)
        hand_opening = math.dist([index.x, index.y], [thumb.x, thumb.y])
        claw_state = "CLAW OPEN" if hand_opening > 0.1 else "CLAW CLOSED"

        # Depth measurement
        hand_size = math.dist([wrist.x, wrist.y], [index.x, index.y])

        # Calibration
        if elapsed < CALIB_FAR_TIME:
            DEPTH_MIN = hand_size if DEPTH_MIN is None else min(DEPTH_MIN, hand_size)

        elif elapsed < CALIB_FAR_TIME + CALIB_CLOSE_TIME:
            DEPTH_MAX = hand_size if DEPTH_MAX is None else max(DEPTH_MAX, hand_size)

        # Depth mapping
        depth_val = 0.7
        if DEPTH_MIN and DEPTH_MAX and DEPTH_MAX - DEPTH_MIN > 1e-4:
            depth_val = (hand_size - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
            depth_val = np.clip(depth_val, 0, 1)
            depth_val = 0.5 + depth_val * 0.5

        # Robot space mapping
        rel_x = pinky.x - neck_ref["x"]
        rel_y = depth_val
        rel_z = pinky.y - neck_ref["shoulder_y"]

        x_scaled = np.clip(-rel_x, -0.7, 0.5)
        y_scaled = np.clip(-rel_y + 0.5, -0.5, 0)
        z_scaled = np.clip(-rel_z + 0.2, 0.25, 1)

        # Move IK (software only)
        frame_count += 1
        if frame_count > 24 * (CALIB_FAR_TIME + CALIB_CLOSE_TIME):
            if frame_count % PLOT_EVERY_N == 0:
                move(x_scaled, y_scaled, z_scaled)

        cv2.putText(
            image,
            f'X:{x_scaled:.2f} Y:{y_scaled:.2f} Z:{z_scaled:.2f}',
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,255),
            2
        )

        cv2.putText(
            image,
            claw_state,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,255),
            2
        )

    cv2.putText(
        image,
        status,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.imshow("Robot Tracking Test (No Arduino)", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
pose.close()
hands.close()