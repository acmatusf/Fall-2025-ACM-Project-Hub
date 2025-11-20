import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the robot arm model
my_chain = ikpy.chain.Chain.from_urdf_file(
    "actual_arm_urdf.urdf", 
    active_links_mask=[False, True, True, True, True, True]
)

# Initial target position and orientation
target_position = [0, 0.2, 0.1]
target_orientation = [-1, 0, 0]

# Calculate inverse kinematics
ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")

print("\n" + "="*60)
print("ROBOTIC ARM INVERSE KINEMATICS VISUALIZATION")
print("="*60)
print("\nTarget Position: ", target_position)
print("Target Orientation: ", target_orientation)
print("\nJoint Angles (degrees):")
for i, angle in enumerate(ik):
    print(f"  Joint {i}: {math.degrees(angle):.2f}°")

# Calculate actual position
computed_position = my_chain.forward_kinematics(ik)
actual_pos = computed_position[:3, 3]
print(f"\nComputed Position: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
print(f"Position Error: {np.linalg.norm(actual_pos - target_position):.4f}m")

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the arm
my_chain.plot(ik, ax, target=target_position)

# Set axis properties
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0, 0.6)
ax.set_xlabel('X (meters)', fontsize=10)
ax.set_ylabel('Y (meters)', fontsize=10)
ax.set_zlabel('Z (meters)', fontsize=10)
ax.set_title('Robotic Arm Configuration', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Add text info on plot
info_text = f"Target: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f})\n"
info_text += f"Actual: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f})"
ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, 
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

print("\n" + "="*60)
print("Close the plot window to continue...")
print("="*60 + "\n")

plt.show()

# Interactive mode - test different positions
print("\n" + "="*60)
print("TEST DIFFERENT POSITIONS")
print("="*60 + "\n")

test_positions = [
    [0, 0.3, 0.2],
    [0.2, 0.2, 0.3],
    [0.1, 0.1, 0.1],
    [-0.1, 0.25, 0.15]
]

for i, pos in enumerate(test_positions, 1):
    print(f"\nTest {i}: Moving to {pos}")
    old_position = ik.copy()
    ik = my_chain.inverse_kinematics(pos, target_orientation, 
                                     orientation_mode="Y", 
                                     initial_position=old_position)
    computed = my_chain.forward_kinematics(ik)
    actual = computed[:3, 3]
    error = np.linalg.norm(actual - pos)
    print(f"  Actual position: [{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}]")
    print(f"  Position error: {error:.4f}m")
    
    # Create a new figure for each test position
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    my_chain.plot(ik, ax, target=pos)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.6)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Test Position {i}: {pos}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    print("Close the window to see the next position...")
    plt.show()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)