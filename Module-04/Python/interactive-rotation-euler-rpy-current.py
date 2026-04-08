import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Explicit Rotation Matrices (Standard math) ---
def Rz(psi):
    """Rotation around Z-axis (Yaw)"""
    c, s = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def Ry(theta):
    """Rotation around Y-axis (Pitch)"""
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])

def Rx(phi):
    """Rotation around X-axis (Roll)"""
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def draw_frame(ax, R_matrix, origin=[0,0,0], scale=1.0, style='solid', label_suffix='', alpha=1.0):
    """
    Helper to draw a coordinate frame based on a rotation matrix.
    R_matrix: 3x3 rotation matrix (columns are axis vectors)
    """
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    
    lines = []
    
    for i in range(3):
        vec = R_matrix[:, i] # Column i is the axis vector
        
        # Line style settings
        ls = '-' if style == 'solid' else '--'
        lw = 2.5 if style == 'solid' else 1.5
        
        # Draw the vector
        # We use quiver for the arrows
        q = ax.quiver(origin[0], origin[1], origin[2], 
                  vec[0], vec[1], vec[2], 
                  color=colors[i], length=scale, linewidth=lw, linestyle=ls, alpha=alpha)
        
        # Label the tip (only for solid frames to avoid clutter)
        if style == 'solid':
            ax.text(vec[0]*scale*1.1, vec[1]*scale*1.1, vec[2]*scale*1.1, 
                    labels[i] + label_suffix, color=colors[i], fontweight='bold', fontsize=10)
        lines.append(q)
    return lines

def update(val):
    ax.cla() # Clear plot
    
    # 1. Get Angles from Sliders
    yaw = s_yaw.val   # Rotation 1: around Z
    pitch = s_pitch.val # Rotation 2: around NEW Y
    roll = s_roll.val  # Rotation 3: around NEW X
    
    # 2. Compute Explicit Matrices (Current Frame = Post-Multiplication)
    # Sequence: Z -> Y' -> X''
    R1 = Rz(yaw)              # 1. Rotate around Global Z
    R2 = R1 @ Ry(pitch)       # 2. Rotate result around ITS OWN Y (Post-multiply)
    R3 = R2 @ Rx(roll)        # 3. Rotate result around ITS OWN X (Post-multiply)
    
    # 3. Visualization
    
    # --- World Frame (Gray, Fixed) ---
    ax.quiver(0,0,0, 1,0,0, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(1.1,0,0, 'W_x', color='k', alpha=0.3)
    ax.quiver(0,0,0, 0,1,0, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(0,1.1,0, 'W_y', color='k', alpha=0.3)
    ax.quiver(0,0,0, 0,0,1, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(0,0,1.1, 'W_z', color='k', alpha=0.3)

    # --- Intermediate Frames (Ghost Frames) ---
    # These visualize the "Current Axis" concept
    
    # Frame 1: After Yaw (Rotated Z)
    if check.get_status()[0]: # If "Show Step 1" checked
        draw_frame(ax, R1, scale=0.8, style='dashed', alpha=0.4)
        # Highlight the axis we are about to rotate around (The Y axis of Frame 1)
        # ax.text(0,0,0, "Step 1: Rotated Z", color='gray', fontsize=8)

    # Frame 2: After Pitch (Rotated Y')
    if check.get_status()[1]: # If "Show Step 2" checked
        draw_frame(ax, R2, scale=0.8, style='dashed', alpha=0.6)
    
    # --- Final Frame (Solid) ---
    draw_frame(ax, R3, scale=1.0, style='solid', label_suffix="''")
    
    # 4. Display Math Info
    info_text = (f"Current Frame Sequence (Z-Y-X):\n"
                 f"1. Yaw (Z): {yaw:.1f}°\n"
                 f"2. Pitch (Y'): {pitch:.1f}°\n"
                 f"3. Roll (X''): {roll:.1f}°\n\n"
                 f"Math: R = Rz({yaw:.0f}) @ Ry({pitch:.0f}) @ Rx({roll:.0f})")
    
    ax.text2D(-0.4, 0.9, info_text, transform=ax.transAxes, family='monospace', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='azure', alpha=0.5))

    # 5. Plot Limits & Settings
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('World X'); ax.set_ylabel('World Y'); ax.set_zlabel('World Z')
    ax.set_title("Euler RPY: Current Frame Rotation")
    ax.set_box_aspect([1,1,1]) 

# --- Setup Figure ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.3)

# Sliders
ax_yaw = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_pitch = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_roll = plt.axes([0.25, 0.10, 0.65, 0.03])

s_yaw = Slider(ax_yaw, '1. Yaw (Z)', -180, 180, valinit=0)
s_pitch = Slider(ax_pitch, '2. Pitch (Y\')', -180, 180, valinit=0)
s_roll = Slider(ax_roll, '3. Roll (X\'\')', -180, 180, valinit=0)

# Checkboxes for Ghosts
ax_check = plt.axes([0.05, 0.4, 0.15, 0.15])
check = CheckButtons(ax_check, ['Show Step 1 (After Z)', 'Show Step 2 (After Y)'], [True, True])

# Update callbacks
s_yaw.on_changed(update)
s_pitch.on_changed(update)
s_roll.on_changed(update)
check.on_clicked(update)

update(0)
plt.show()
