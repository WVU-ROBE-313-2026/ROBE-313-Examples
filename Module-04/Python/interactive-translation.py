import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from, plot_transform

# 1. Setup the Figure
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35, left=0.1) 

def update(val):
    ax.cla() 
    
    # 2. Get Rotation Euler Angles (Intrinsic Z-Y-X)
    alpha, beta, gamma = s_yaw.val, s_pitch.val, s_roll.val
    R = matrix_from_euler(np.deg2rad([alpha, beta, gamma]), 0, 1, 2, extrinsic=False)
    
    # 3. Get Translation Components
    tx, ty, tz = s_tx.val, s_ty.val, s_tz.val
    t = np.array([tx, ty, tz])
    
    # 4. Create the 4x4 Homogeneous Transformation Matrix T
    T_AB = transform_from(R, t)
    
    # 5. Draw Frames
    # Frame A (World)
    plot_transform(ax=ax, A2B=np.eye(4), s=0.5, lw=2, name="Frame {a}")
    
    # Frame B (Body/Transformed)
    plot_transform(ax=ax, A2B=T_AB, s=0.8, lw=3, name="Frame {b}")
    
    # 6. Projection of the Origin (Visualizing Translation)
    # Draw drop-lines from the origin of {b} to the XY plane and axes
    # Line from (tx, ty, tz) to (tx, ty, 0)
    ax.plot([tx, tx], [ty, ty], [0, tz], 'k--', alpha=0.4)
    # Line from (tx, ty, 0) to (tx, 0, 0)
    ax.plot([tx, tx], [0, ty], [0, 0], 'k:', alpha=0.3)
    # Line from (tx, ty, 0) to (0, ty, 0)
    ax.plot([0, tx], [ty, ty], [0, 0], 'k:', alpha=0.3)
    
    # Highlight the origin point of {b}
    ax.scatter(tx, ty, tz, color='black', s=20)

    # 7. Display the Matrix T
    matrix_text = (
        f"T = [ R | p ]\n"
        f"{np.round(T_AB, 2)}"
    )
    ax.text2D(-0.2, 0.90, matrix_text, transform=ax.transAxes, 
              family='monospace', fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set limits
    ax.set_xlim([-1, 3]); ax.set_ylim([-1, 3]); ax.set_zlim([-1, 3])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Homogeneous Transformation: $\\mathbf{p}_a = \\mathbf{T}_a^b \\mathbf{p}_b$")
    fig.canvas.draw_idle()

# --- GUI Layout ---
# Rotation Sliders
ax_yaw   = plt.axes([0.2, 0.22, 0.25, 0.03])
ax_pitch = plt.axes([0.2, 0.17, 0.25, 0.03])
ax_roll  = plt.axes([0.2, 0.12, 0.25, 0.03])

# Translation Sliders
ax_tx = plt.axes([0.6, 0.22, 0.25, 0.03])
ax_ty = plt.axes([0.6, 0.17, 0.25, 0.03])
ax_tz = plt.axes([0.6, 0.12, 0.25, 0.03])

s_yaw   = Slider(ax_yaw,   'Yaw (Z)',   -180, 180, valinit=0)
s_pitch = Slider(ax_pitch, 'Pitch (Y)', -180, 180, valinit=0)
s_roll  = Slider(ax_roll,  'Roll (X)',  -180, 180, valinit=0)

s_tx = Slider(ax_tx, 'X pos', -1, 2, valinit=1.0)
s_ty = Slider(ax_ty, 'Y pos', -1, 2, valinit=1.0)
s_tz = Slider(ax_tz, 'Z pos', -1, 2, valinit=1.0)

for s in [s_yaw, s_pitch, s_roll, s_tx, s_ty, s_tz]:
    s.on_changed(update)

update(0)
plt.show()
