"""#!/usr/bin/env python
# coding: utf-8"""

# We start by importing all our libraries

# In[1]:


import ikpy.chain
#import ikpy.utils.plot as plot_utils

import numpy as np
import math

import serial
import time
import serial.tools.list_ports


# Now, we can import our robot arm model from the URDF file. The first link is the link between the desk and the base, which doesn't move, so we set it to inactive

# In[2]:


my_chain = ikpy.chain.Chain.from_urdf_file("Hardware/actual_arm_urdf.urdf", active_links_mask=[False, True, True, True, True, True])


# And set the target position and orientation of the arm

# In[3]:


target_position = [0, 0.2, 0.1]

target_orientation = [-1, 0, 0]


# It's now just one call to work out the inverse kinematics for that position. Again, the first angle is of the inactive joint between the desk and the base so is always 0

# In[4]:


ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")
print("The angles of each joints are: ", list(map(lambda r: math.degrees(r), ik.tolist())))


# We can see the actual position our robot will move to. This may be different to target_position as the arm may not be physically able to reach that position

# In[5]:


computed_position = my_chain.forward_kinematics(ik)
print("Computed position:", [f"{val:.2f}" for val in computed_position[:3, 3]])


# Now let's visualize what our arm looks (remember to check which directions correspond with which axis (x, y, or z) to accurately move the arm)

# In[6]:


"""
%matplotlib qt
import matplotlib.pyplot as plt
fig, ax = plot_utils.init_3d_figure()
fig.set_figheight(18)  
fig.set_figwidth(26)  
my_chain.plot(ik, ax, target=target_position)
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
ax.set_zlim(0, 0.6)
plt.ion()
"""


# Then we'll find the Arduino port to setup Serial connection to send the results of ik to move the arm

# In[7]:


# before we debug the functions below this part we have to reset Serial connection
# i.e. set ser = None, debug the functions, then run the next cell to setup ser again
ser = None


# In[8]:


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


# Now we'll create a function to actually send the ik results to Arduino to move the arm

# In[9]:


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
        neutral_servos = [int(math.degrees(0.2)),                # Gripper
                          rad_to_motor_degree(0.0, -1.92, 1.22), # Wrist 2
                          rad_to_motor_degree(0.0, -1.92, 0.7),  # Wrist 1
                          rad_to_motor_degree(0.0, -1.39, 1.57), # Elbow
                          rad_to_motor_degree(0.0, -1.39, 1.57)  # Shoulders
                         ]
        _send_packet(neutral_servos, base_current)
        time.sleep(2.5) # give the servos time to move to their neutral position

        # keep servos neutral, move base if needed
        if base_target != base_current:
            _send_packet(neutral_servos, base_target)
            time.sleep(3.5) # give the base time to move

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


# Now we'll wrap up some of these calls into a couple of functions. Calling move(x,y,z) will move us to the new coordinates and update the plot.
# It's worth noting here that when we call inverse_kinematics, we pass in the old position (joint angles) as initial_position so IKPY find the nearest solution to our current position.

# In[10]:


def doIK():
    global ik
    old_position = ik.copy()
    ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode=None, initial_position=old_position)
    print("The angles of each joints are: ", list(map(lambda r: math.degrees(r), ik.tolist())))

"""
def updatePlot():
    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    my_chain.plot(ik, ax, target=target_position)
    fig.canvas.draw_idle()
"""

def move(x, y, z):
    global target_position
    target_position = [x, y, z]
    doIK()
    #updatePlot()
    send_all_motors(0.2, ik[5].item(), ik[4].item(), ik[3].item(), ik[2].item(), ik[1].item())


# We should now be able to move our visualized arm with a call to move(x,y,z)

# In[19]:

move(0.2, 0, 0.8)
time.sleep(5)
move(-0.2, 0.1, 0.8)
time.sleep(5)
move(0, -0.1, 0.8)


# In[46]:


#send_all_motors(0.2, 0, 0, 0, 0, 0)


# In[13]:


#send_all_motors(0.2, 0, 0, 0, -0.3, 4.71)


# In[21]:


# Run this in a notebook cell to convert this jupyter notebook to a python file
#get_ipython().system('jupyter nbconvert --to script Actual_Inverse_Kinematics.ipynb')


# In[ ]:




