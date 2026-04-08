import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Math Functions (Rodrigues / Exponential Map) ---

def skew(v):
    """Returns the 3x3 skew-symmetric matrix of vector v."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])

def rodrigues_formula(k, theta_deg):
    """
    Computes Rotation Matrix R using Rodrigues' Formula.
    R = I + sin(t)K + (1-cos(t))K^2
    where K is the skew-symmetric matrix of the unit vector k.
    """
    theta_rad = np.radians(theta_deg)
    
    # Handle zero vector case
    norm = np.linalg.norm(k)
    if norm < 1e-6:
        return np.eye(3), np.zeros(3) # Return Identity if axis is 0
    
    u = k / norm # Normalize to get unit axis
    K = skew(u)
    
    # The Formula
    R = np.eye(3) + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * (K @ K)
    
    return R, u

def draw_frame(ax, R_matrix, origin=[0,0,0], scale=1.0, label_prefix='B'):
    """Draws the frame defined by R_matrix columns."""
    colors = ['r', 'g', 'b']
    labels = ['_x', '_y', '_z']
    
    for i in range(3):
        vec = R_matrix[:, i]
        ax.quiver(origin[0], origin[1], origin[2], 
                  vec[0], vec[1], vec[2], 
                  color=colors[i], length=scale, linewidth=3)
        ax.text(vec[0]*scale*1.1, vec[1]*scale*1.1, vec[2]*scale*1.1, 
                label_prefix + labels[i], color=colors[i], fontweight='bold')

def update(val):
    ax.cla() # Clear plot
    
    # 1. Get Inputs
    kx, ky, kz = s_kx.val, s_ky.val, s_kz.val
    theta = s_theta.val
    
    k_raw = np.array([kx, ky, kz])
    
    # 2. Compute Rotation
    R_matrix, u_unit = rodrigues_formula(k_raw, theta)
    
    # 3. Visualization
    
    # --- World Frame (Fixed) ---
    ax.quiver(0,0,0, 1,0,0, color='k', alpha=0.1, length=1.2)
    ax.quiver(0,0,0, 0,1,0, color='k', alpha=0.1, length=1.2)
    ax.quiver(0,0,0, 0,0,1, color='k', alpha=0.1, length=1.2)
    
    # --- Rotation Axis (The "k" vector) ---
    # We draw this distinctively (Black Dashed)
    if np.linalg.norm(k_raw) > 1e-6:
        # Draw a long axis line to visualize it as an infinite axis
        ax.quiver(0,0,0, u_unit[0], u_unit[1], u_unit[2], 
                  color='k', linestyle='--', linewidth=1.5, length=1.5, arrow_length_ratio=0.1)
        ax.quiver(0,0,0, -u_unit[0], -u_unit[1], -u_unit[2], 
                  color='k', linestyle='--', linewidth=1.5, length=1.5, arrow_length_ratio=0.0)
        ax.text(u_unit[0]*1.6, u_unit[1]*1.6, u_unit[2]*1.6, "Axis k", color='k', fontweight='bold')

    # --- Rotated Body Frame ---
    draw_frame(ax, R_matrix)
    
    # 4. Display Math Info (The Exponential Map context)
    
    # Format the Skew Matrix (S = theta * [u]x)
    # The exponential coordinate vector is w = theta * u
    w = np.radians(theta) * u_unit if np.linalg.norm(k_raw) > 1e-6 else np.zeros(3)
    
    info_text = (f"Angle-Axis Representation:\n"
                 f"Axis k = [{u_unit[0]:.2f}, {u_unit[1]:.2f}, {u_unit[2]:.2f}] (Unit)\n"
                 f"Angle θ = {theta:.1f}°\n\n"
                 f"Exponential Map (Rodrigues):\n"
                 f"R = I + sin(θ)K + (1-cos(θ))K²\n"
                 f"where K = [k]ₓ (Skew-Symmetric)\n\n"
                 f"Exp Coordinate ω = θ·k:\n"
                 f"[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]")
    
    ax.text2D(-0.8, 0.5, info_text, transform=ax.transAxes, family='monospace', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='azure', alpha=0.5))

    # 5. Plot Settings
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(r"Exponential Map: $R = e^{[k]_\times \theta}$")
    ax.set_box_aspect([1,1,1]) 

def reset(event):
    s_kx.reset()
    s_ky.reset()
    s_kz.reset()
    s_theta.reset()

# --- Setup Figure ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.35)

# Sliders
ax_kx = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_ky = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_kz = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta = plt.axes([0.25, 0.10, 0.65, 0.03])

s_kx = Slider(ax_kx, 'Axis X (kx)', -1, 1, valinit=0)
s_ky = Slider(ax_ky, 'Axis Y (ky)', -1, 1, valinit=0)
s_kz = Slider(ax_kz, 'Axis Z (kz)', -1, 1, valinit=1) # Default Z-axis
s_theta = Slider(ax_theta, 'Angle θ (deg)', -180, 180, valinit=0)

# Reset Button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
button.on_clicked(reset)

# Update callbacks
s_kx.on_changed(update)
s_ky.on_changed(update)
s_kz.on_changed(update)
s_theta.on_changed(update)

update(0)
plt.show()
