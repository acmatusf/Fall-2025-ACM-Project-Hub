# We start by importing all our libraries
import ikpy.chain
import numpy as np
import math
import serial
import time
import serial.tools.list_ports
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

# Now, we can import our robot arm model from the URDF file. 
# The first link is the link between the desk and the base, which doesn't move, so we set it to inactive

my_chain = ikpy.chain.Chain.from_urdf_file(
    "Hardware/actual_arm_urdf.urdf",
    active_links_mask=[False, True, True, True, True, True]
)

# And set the target position and orientation of the arm
target_position = [0, 0.2, 0.1]
target_orientation = [-1, 0, 0]

# It's now just one call to work out the inverse kinematics for that position. 
ik = my_chain.inverse_kinematics(
    target_position,
    target_orientation,
    orientation_mode="Y"
)

# -------------------------
# Declaring the claw state
# -------------------------
claw_state = "HAND NOT DETECTED"

# -------------------------
# Arduino Setup
# -------------------------

# Then we'll find the Arduino port to setup Serial connection to send the results of ik to move the arm
def find_arduino():
    # auto-detect Arduino COM port
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if ("Arduino" in p.description) or ("CH340" in p.description) or ("USB Serial" in p.description):
            return p.device
    return None
# try to find Arduino
arduino_port = find_arduino()
if not arduino_port:
    print("Error", "No Arduino detected. Please connect your board and restart.")
    exit()
try:
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)  # wait for Arduino to reset
except Exception as e:
    print("Connection Error", f"Could not open {arduino_port}:\n{e}")
    exit()


# -------------------------
# Motor Control
# -------------------------

# convert ik results (in radians) to motor degree angles
def rad_to_motor_degree(rad, rad_min, rad_max, motor_min=0, motor_max=180):
    # clamp for safety
    rad = max(min(rad, rad_max), rad_min)
    # normalize 0 - 1
    normalized = (rad - rad_min) / (rad_max - rad_min)
    # scale to motor range
    return int(motor_min + normalized * (motor_max - motor_min))

"""
angle1 = Gripper
angle2 = Wrist 2
angle3 = Wrist 1
angle4 = Elbow
angle5 = Shoulders
value = Stepper
"""
# send motor values to Arduino for parsing through Serial
def _send_packet(servo_vals, stepper_val):
    data = " ".join(map(str, servo_vals)) + " " + str(stepper_val) + "\n"
    ser.write(data.encode())

last_sent = 0
last_base_value = 0.0 # radians
def send_all_motors(angle1, angle2, angle3, angle4, angle5, value):
    global last_sent, last_base_value
    # slightly delay motors to prevent jagged movement
    now = time.time()
    if now - last_sent < 0.1:
        return

    try:
        # convert base angles
        base_current = int(rad_to_motor_degree(last_base_value, 0, 4.71, 0, 270))
        base_target  = int(rad_to_motor_degree(value, 0, 4.71, 0, 270))
    except ValueError:
        return

    # if the arm is not neutralized (i.e. all servos are returned to their neutral position at 0 degrees), the weight is redistributed
    # this causes the stepper motor to require more torque to move, resulting in incorrect base position
    # e.g. base only rotates 180 degrees instead of the desired 270 degrees
    # the solution is to return the arm to its neutral position, move the base, then move the arm to its target position
    # note that each servo has a limited physical range of rotation depending on assembly of the arm, 
    # therefore rad_min and rad_max may vary (equal to lower and upper limits of joints specified in the urdf file)
    try:
        # neutralize servos, don't move base
        neutral_servos = [int(math.degrees(0.0)),                # Gripper
                          rad_to_motor_degree(0.0, -1.92, 1.22), # Wrist 2
                          rad_to_motor_degree(0.0, -1.92, 0.7),  # Wrist 1
                          rad_to_motor_degree(0.0, -1.39, 1.57), # Elbow
                          rad_to_motor_degree(0.0, -1.39, 1.57)  # Shoulders
                         ]
        _send_packet(neutral_servos, base_current)
        time.sleep(2) # give the servos time to move to their neutral position

        # keep servos neutral, move base if needed
        if base_target != base_current:
            _send_packet(neutral_servos, base_target)
            time.sleep(3) # give the base time to move

        # move servos to their target position, don't move base
        target_servos = [int(math.degrees(angle1)),                # Gripper
                         rad_to_motor_degree(angle2, -1.92, 1.22), # Wrist 2
                         rad_to_motor_degree(angle3, -1.92, 0.7),  # Wrist 1
                         rad_to_motor_degree(angle4, -1.39, 1.57), # Elbow
                         rad_to_motor_degree(angle5, -1.39, 1.57)  # Shoulders
                        ]
        _send_packet(target_servos, base_target)

        # update the base's current position and the last time the arm moved
        last_base_value = value
        last_sent = now
    except serial.SerialException:
        print("Connection Lost", "Lost connection to Arduino.")

# moves only the gripper without delay to open/close the claw
def move_gripper(angle_deg):
    angle_deg = int(max(0, min(180, angle_deg)))
    ser.write(f"G {angle_deg}\n".encode())


# -------------------------
# IK Wrapper
# -------------------------

# Now we'll wrap up some of these calls into a couple of functions. 
# Calling move(x,y,z) will move us to the new coordinates and update the plot.
# It's worth noting here that when we call inverse_kinematics, we pass in the old position (joint angles) 
# as initial_position so IKPY find the nearest solution to our current position.
def doIK():
    global ik
    old_position = ik.copy()
    ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode=None, initial_position=old_position)
    print("The angles of each joints are: ", list(map(lambda r: math.degrees(r), ik.tolist())))

def move(x, y, z):
    global target_position
    target_position = [x, y, z]
    doIK()
    send_all_motors(
        0.0,
        ik[5].item(),
        ik[4].item(),
        ik[3].item(),
        ik[2].item(),
        ik[1].item()
    )


# reset all motors and give the program/arm delay time before working
send_all_motors(0.2, 0, 0, 0, 0, 0)
time.sleep(3)


# -------------------------
# Camera Loop
# -------------------------

frame_count = 0
PLOT_EVERY_N = 200

frame_count_claw = 0
PLOT_EVERY_N_CLAW = 72

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

    # -------------------------
    # Neck Reference
    # -------------------------
    neck_ref = None

    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        neck_ref = {
            "x": (ls.x + rs.x) / 2,
            "z": (ls.y + rs.y) / 2,   # shoulder height reference
            "shoulder_y": (ls.y + rs.y) / 2
        }

    # -------------------------
    # Hand Tracking
    # -------------------------

    if hand_results.multi_hand_landmarks and neck_ref:
        hand = hand_results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            image,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        lm = hand.landmark

        wrist = lm[0]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]
        pinky = lm[20]

        # -------------------------
        # Claw setup
        # -------------------------

        thumb = lm[4]
        thumb_x = thumb.x
        thumb_y = thumb.y

        hand_opening = math.dist(
            [index.x, index.y],
            [thumb_x, thumb_y]
        )

        if hand_opening > 0.1:
            claw_state = "CLAW OPEN"
        else:
            claw_state = "CLAW CLOSED"

        # -------------------------
        # Hand Size (Depth)
        # -------------------------

        hand_size = math.dist(
            [wrist.x, wrist.y],
            [index.x, index.y]
        )

        # -------------------------
        # Depth Calibration - FAR AWAY
        # -------------------------
        if elapsed < CALIB_FAR_TIME:
            if DEPTH_MIN is None or hand_size < DEPTH_MIN:
                DEPTH_MIN = hand_size
        # -------------------------
        # Depth Calibration - CLOSE
        # -------------------------
        elif elapsed < CALIB_FAR_TIME + CALIB_CLOSE_TIME:
            if DEPTH_MAX is None or hand_size > DEPTH_MAX:
                DEPTH_MAX = hand_size

        # -------------------------
        # Depth Mapping
        # -------------------------

        depth_val = 0.7
        if DEPTH_MIN and DEPTH_MAX:
            if DEPTH_MAX - DEPTH_MIN > 1e-4:
                depth_val = (hand_size - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
                depth_val = np.clip(depth_val, 0, 1)
                depth_val = 0.5 + depth_val * 0.5

        # -------------------------
        # Robot-space Position (Fix A)
        # -------------------------

        # Horizontal movement: wrist relative to shoulders
        rel_x = pinky.x - neck_ref["x"]

        # Depth: from calibrated hand size ONLY
        rel_y = depth_val

        # Height: wrist height relative to shoulders
        # (this is the magic line)
        rel_z = pinky.y - neck_ref["shoulder_y"]

        x_scaled = -rel_x
        y_scaled = -rel_y + 0.5
        z_scaled = -rel_z + 0.2

        x_scaled = np.clip(x_scaled, -0.7, 0.5)
        y_scaled = np.clip(y_scaled, -0.5, 0)
        z_scaled = np.clip(z_scaled, 0.25, 1)

        # -------------------------
        # Move Robot
        # -------------------------

        frame_count += 1
        if frame_count > 24 * (CALIB_FAR_TIME + CALIB_CLOSE_TIME) and frame_count % PLOT_EVERY_N_CLAW == 0:
            if claw_state == "CLAW OPEN":
                # Open claw command
                move_gripper(180)
                print("Opening claw\n")
            else:
                # Close claw command
                move_gripper(0)
                print("Closing claw\n")
        elif frame_count > 24 * (CALIB_FAR_TIME + CALIB_CLOSE_TIME) and frame_count % PLOT_EVERY_N == 0:
            move(x_scaled, y_scaled, z_scaled)
            print(
                f"Target X:{x_scaled:.2f} "
                f"Y:{y_scaled:.2f} "
                f"Z:{z_scaled:.2f}"
            )

        # -------------------------
        # Depth Display
        # -------------------------
        cv2.putText(
            image,
            f'X:{x_scaled:.2f} Y:{y_scaled:.2f} Z:{z_scaled:.2f}',
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,255),
            2
        )

    # -------------------------
    # Depth and Claw Status Display
    # -------------------------
    cv2.putText(
        image,
        f'{status} {claw_state}',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.namedWindow("Robot Control", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Robot Control', cv2.WND_PROP_FULLSCREEN, 0)
    cv2.imshow("Robot Control", image)

    if cv2.waitKey(1) & 0xFF == ord("q"): # quit the program by pressing q
        break

    
# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()

pose.close()
hands.close()
send_all_motors(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)