import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Explicit Rotation Matrices ---
def Rz(angle):
    """Rotation around Z-axis"""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def Ry(angle):
    """Rotation around Y-axis"""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])

def draw_frame(ax, R_matrix, origin=[0,0,0], scale=1.0, style='solid', label_suffix='', alpha=1.0):
    """Helper to draw a coordinate frame based on a rotation matrix."""
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    
    for i in range(3):
        vec = R_matrix[:, i] # Column i is the axis vector
        
        # Line style settings
        ls = '-' if style == 'solid' else '--'
        lw = 2.5 if style == 'solid' else 1.5
        
        # Draw the vector using quiver
        ax.quiver(origin[0], origin[1], origin[2], 
                  vec[0], vec[1], vec[2], 
                  color=colors[i], length=scale, linewidth=lw, linestyle=ls, alpha=alpha)
        
        # Label the tip (only for visible frames)
        if style == 'solid' or alpha > 0.4:
            ax.text(vec[0]*scale*1.1, vec[1]*scale*1.1, vec[2]*scale*1.1, 
                    labels[i] + label_suffix, color=colors[i], fontweight='bold', fontsize=9)

def update(val):
    ax.cla() # Clear plot
    
    # 1. Get Angles from Sliders
    phi = s_phi.val     # Rotation 1: around Z
    theta = s_theta.val # Rotation 2: around NEW Y
    psi = s_psi.val     # Rotation 3: around NEW Z
    
    # 2. Compute Explicit Matrices (Current Frame = Post-Multiplication)
    # Sequence: Z -> Y' -> Z''
    
    # Step 1: Rotate around Global Z
    R1 = Rz(phi)
    
    # Step 2: Rotate result around ITS OWN Y (Post-multiply)
    R2 = R1 @ Ry(theta)
    
    # Step 3: Rotate result around ITS OWN Z (Post-multiply)
    R3 = R2 @ Rz(psi)
    
    # 3. Visualization
    
    # --- World Frame (Gray, Fixed) ---
    ax.quiver(0,0,0, 1,0,0, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(1.1,0,0, 'W_x', color='k', alpha=0.3)
    ax.quiver(0,0,0, 0,1,0, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(0,1.1,0, 'W_y', color='k', alpha=0.3)
    ax.quiver(0,0,0, 0,0,1, color='k', alpha=0.1, arrow_length_ratio=0.1)
    ax.text(0,0,1.1, 'W_z', color='k', alpha=0.3)

    # --- Intermediate Frames (Ghost Frames) ---
    
    # Frame 1: After Phi (Rotated Z)
    # This visualizes the Y' axis we are about to rotate around next
    if check.get_status()[0]: 
        draw_frame(ax, R1, scale=0.8, style='dashed', label_suffix="'", alpha=0.4)

    # Frame 2: After Theta (Rotated Y')
    # This visualizes the Z'' axis we are about to rotate around last
    if check.get_status()[1]: 
        draw_frame(ax, R2, scale=0.8, style='dashed', label_suffix="''", alpha=0.6)
    
    # --- Final Frame (Solid) ---
    draw_frame(ax, R3, scale=1.0, style='solid', label_suffix="'''")
    
    # 4. Display Math Info
    info_text = (f"Current Frame Sequence (Z-Y-Z):\n"
                 f"1. Phi (Z):   {phi:.1f}°\n"
                 f"2. Theta (Y'): {theta:.1f}°\n"
                 f"3. Psi (Z''):  {psi:.1f}°\n\n"
                 f"Math: R = Rz({phi:.0f}) @ Ry({theta:.0f}) @ Rz({psi:.0f})")
    
    ax.text2D(-0.4, 0.9, info_text, transform=ax.transAxes, family='monospace', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # 5. Plot Limits & Settings
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('World X'); ax.set_ylabel('World Y'); ax.set_zlabel('World Z')
    ax.set_title("Euler ZYZ: Current Frame Rotation")
    ax.set_box_aspect([1,1,1]) 

# --- Setup Figure ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.3)

# Sliders
ax_phi = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_theta = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_psi = plt.axes([0.25, 0.10, 0.65, 0.03])

s_phi = Slider(ax_phi, '1. Phi (Z)', -180, 180, valinit=0)
s_theta = Slider(ax_theta, '2. Theta (Y\')', -180, 180, valinit=0)
s_psi = Slider(ax_psi, '3. Psi (Z\'\')', -180, 180, valinit=0)

# Checkboxes for Ghosts
ax_check = plt.axes([0.05, 0.4, 0.15, 0.15])
check = CheckButtons(ax_check, ['Show Step 1 (After Z)', 'Show Step 2 (After Y\')'], [True, True])

# Update callbacks
s_phi.on_changed(update)
s_theta.on_changed(update)
s_psi.on_changed(update)
check.on_clicked(update)

update(0)
plt.show()