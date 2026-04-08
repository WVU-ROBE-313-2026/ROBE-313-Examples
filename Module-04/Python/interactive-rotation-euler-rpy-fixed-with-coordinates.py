import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R

def draw_projections(ax, vec):
    """
    Draws dotted lines from the tip of a vector to the X, Y, and Z axes
    to visualize its components in the Global Frame.
    vec: The 3D coordinates of the vector tip (since origin is 0,0,0)
    """
    x, y, z = vec
    
    # 1. Drop line to X-axis (Red dotted)
    # Draw line from tip to (x, 0, 0)
    ax.plot([x, x], [y, 0], [z, 0], 'r:', alpha=0.5, linewidth=1)
    # Highlight the point on the axis
    ax.scatter([x], [0], [0], color='r', s=20)
    
    # 2. Drop line to Y-axis (Green dotted)
    ax.plot([x, 0], [y, y], [z, 0], 'g:', alpha=0.5, linewidth=1)
    ax.scatter([0], [y], [0], color='g', s=20)
    
    # 3. Drop line to Z-axis (Blue dotted)
    ax.plot([x, 0], [y, 0], [z, z], 'b:', alpha=0.5, linewidth=1)
    ax.scatter([0], [0], [z], color='b', s=20)

def update(val):
    ax.cla() # Clear plot
    
    # 1. Get Euler Angles
    yaw, pitch, roll = s_yaw.val, s_pitch.val, s_roll.val
    
    # 2. Compute Rotation
    r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    # r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    rot_matrix = r.as_matrix()
    
    # 3. Visualize Frames
    
    # --- Draw Fixed World Frame (Faint) ---
    # We draw this first so it's in the background. Thin lines.
    origin = np.zeros(3)
    ax.quiver(0,0,0, 1,0,0, color='r', alpha=0.2, linestyle='--', length=1.2) # World X
    ax.quiver(0,0,0, 0,1,0, color='g', alpha=0.2, linestyle='--', length=1.2) # World Y
    ax.quiver(0,0,0, 0,0,1, color='b', alpha=0.2, linestyle='--', length=1.2) # World Z
    
    # --- Draw Mobile Body Frame (Bold) ---
    # The columns of rot_matrix ARE the Body axes expressed in World Frame
    colors = ['r', 'g', 'b']
    labels = ['B_x', 'B_y', 'B_z']
    
    for i in range(3):
        vec = rot_matrix[:, i] # Column i
        
        # Draw the vector (Modulus 1.0)
        ax.quiver(0,0,0, vec[0], vec[1], vec[2], 
                  color=colors[i], length=1.0, linewidth=2.5, normalize=True)
        
        # Label the tip
        ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, labels[i], 
                color=colors[i], fontweight='bold')
        
        # --- THE TEACHING MOMENT: PROJECTIONS FOR B_X ONLY ---
        # We only draw projection lines for the Red Axis (B_x) to avoid clutter
        # and to focus on the "First Column" concept.
        if i == 0:
            draw_projections(ax, vec)
            
            # Add value labels directly on the World Axes
            ax.text(vec[0], 0, 0, f"{vec[0]:.2f}", color='r', fontsize=8, ha='center', va='bottom')
            ax.text(0, vec[1], 0, f"{vec[1]:.2f}", color='g', fontsize=8, ha='center', va='bottom')
            ax.text(0, 0, vec[2], f"{vec[2]:.2f}", color='b', fontsize=8, ha='center', va='bottom')

    # 4. Update Matrix Text
    col1 = rot_matrix[:, 0]
    
    # We format the matrix to look like a standard mathematical object
    matrix_str = (f"Rotation Matrix R:\n"
                  f"[[{rot_matrix[0,0]:5.2f}, {rot_matrix[0,1]:5.2f}, {rot_matrix[0,2]:5.2f}]\n"
                  f" [{rot_matrix[1,0]:5.2f}, {rot_matrix[1,1]:5.2f}, {rot_matrix[1,2]:5.2f}]\n"
                  f" [{rot_matrix[2,0]:5.2f}, {rot_matrix[2,1]:5.2f}, {rot_matrix[2,2]:5.2f}]]")
    
    # Helper text to explain the projection
    explanation = (f"Observation:\n"
                   f"The coordinates of B_x (Red Arrow)\n"
                   f"on the World Axes are:\n"
                   f"X: {col1[0]:.2f}\n"
                   f"Y: {col1[1]:.2f}\n"
                   f"Z: {col1[2]:.2f}\n\n"
                   f"This matches Column 1 exactly!")

    # Display Text
    ax.text2D(-0.4, 0.95, matrix_str, transform=ax.transAxes, family='monospace', 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
              
    ax.text2D(-0.4, 0.65, explanation, transform=ax.transAxes, family='sans-serif', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

    # 5. Plot Limits & Settings
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('World X'); ax.set_ylabel('World Y'); ax.set_zlabel('World Z')
    ax.set_title("Interpretation of Rotation Matrix Columns")
    
    # Force equal aspect ratio to ensure length 1 looks like length 1
    ax.set_box_aspect([1,1,1]) 

# --- Setup Figure ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Sliders
ax_yaw = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_pitch = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_roll = plt.axes([0.25, 0.05, 0.65, 0.03])

s_yaw = Slider(ax_yaw, 'Yaw (Z)', -180, 180, valinit=0)
s_pitch = Slider(ax_pitch, 'Pitch (Y)', -180, 180, valinit=0)
s_roll = Slider(ax_roll, 'Roll (X)', -180, 180, valinit=0)

s_yaw.on_changed(update)
s_pitch.on_changed(update)
s_roll.on_changed(update)

update(0)
plt.show()
