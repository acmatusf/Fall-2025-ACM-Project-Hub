# Fall-2025-ACM-Project-Hub
## Project Hub Overview

For this project hub, we built a gesture controlled robotic arm, which tracks your hand using your computer's camera and commands the arm to move accordingly. In this project, we used many technologies including computer vision/machine learning, inverse kinematics, Arduino programming, and a bit of 3D printing, CAD design, and circuit design.

This project was designed to help students learn by doing. Instead of traditional workshops, our role is to guide you as you collaborate through Discord and GitHub. Additionally, by working on this project together, you’ll also gain hands-on experience in coding, teamwork, and making documentation.

Our end goal is to showcase the project in a final event where you can present your work, since it’s about building a project that demonstrates your skills and looks great on your resume, boosting your internship chances and paving the way for future Project Hubs. This year we got the opportunity to present at the ACM AI Convention 2026, the flagship event hosted by the ACM chapter at USF (University of South Florida). This is where many startups, companies, research labs, and student organizations around Tampa will come and showcase their best projects (and for some, job openings as well) utilizing AI to spark further interest and discussion about AI.
<img src="AI_Convention_2026_poster.jpg" width="300">
<img src="AI_Convention_2026_project_presentation.jpg" width="291">

## How It Works
At a high level, the logic follows like this:
1. The camera from your computer tracks your hand, specifically your pinky finger, using a ML/CV model and uses your neck as the reference to determine the finger's relative coordinates. It also measures the distance between the thumb and index finger to determine whether or not the claw/gripper should be opened.
2. After a few seconds (modifiable within the code), the camera will record the coordinates and freezes for some time, which means it is telling the robotic arm to move accordingly.
3. The recorded coordinates will be passed through an inverse kinematics function, which would basically tell the Arduino which motors to rotate and rotate by how much to reach that position in the real world.
4. Once the recorded coordinates data have been passed through the inverse kinematics function, it is sent to Arduino through Serial connection. The Arduino receives them and control the motors, rotating each exact ones by the exact amount to move the robotic arm to match the user's (you) hand position.
5. Due to certain physical limitations (which will be explained more within the code), the arm will take some time to move to its designated position, thus resulting in the occasional frame freezes above. The logic for opening and closing the claw/gripper follows similarly, but with a lower wait time and no delay.

## How To Build And Run The Robotic Arm
Step 1: 

## How To Run The Software
1. Connect the arduino to your computer via USB cable. Then, open the /Software folder and open the "Complete_arm_control_code.py" file, which is the code of our robotic arm operation.
2. Run "pip install -r requirements.txt" in the terminal. This will install every library needed for the complete operation of the robot listed in the requirements.txt file.
3. Now to start the operation of the robotic arm, you can run the "Complete_arm_control_code.py" file. When it launches, two instructions will appear on the top of the camera frame: "MOVE HAND FAR" and "MOVE HAND CLOSE". This is the calibration stage of the software. So when prompted, move your hand as far away from the camera you can without moving your shoulder excessively, same thing with close to the camera. After that, "CALIBRATION DONE" should appear on the screen. If it is felt that the calibration wasn't done correctly, it is reccomended to close and run the software again.
4. With the software calibrated, the arm should be operational if correctly setted up: The code works by moving the robotic arm to the user's hand position relative to their shoulder. Also, by tracking if the index and thumb fingers of the user are touching the software defines the claw as open or closed.
5. The software takes some time to move the robot, and it freezes everytime it is happening so the robot has its needed time to move. In case the freezes are too frequent or too spaced away, modify lines 223 and 226 PLOT_EVERY_N and PLOT_EVERY_N_CLAW.
