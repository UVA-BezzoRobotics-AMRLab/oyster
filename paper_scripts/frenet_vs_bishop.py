import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
# CONFIGURATION — 2D Navigation Mode
# ==============================================================================

FRAME_MODE = "RMF"  # "RMF" or "FS" (FS in 2D is the standard Signed Curvature frame)
# FRAME_MODE = "FS"  # "RMF" or "FS" (FS in 2D is the standard Signed Curvature frame)

N_CURVE  = 300      # dense curve points
N_FRAMES = 15       # sparse frame triad locations
N_TUBE   = 300      # dense corridor points

# In 2D, these represent the distance to the left and right boundaries
# We can make them unequal to show the "flip" dramatically in FS mode
CORRIDOR_WIDTH_LEFT  = 0.4 
CORRIDOR_WIDTH_RIGHT = 0.2 

FRAME_LENGTH  = 0.3
HALF_WINDOW   = 1.5 # radians around pi

# Palette
C_CURVE = "#1C1C2E"
C_CORRIDOR = "#4A90D9"
C_T   = "#E05252"
C_E1  = "#52A852"  # Normal vector
BG    = "white"

# ==============================================================================
# 2D Curve definition (S-curve in XY plane)
# ==============================================================================

def curve(t):
    x = t
    y = np.sin(t)
    return np.stack([x, y], axis=-1)

def curve_d1(t):
    dx = np.ones_like(t)
    dy = np.cos(t)
    return np.stack([dx, dy], axis=-1)

def curve_d2(t):
    dx = np.zeros_like(t)
    dy = -np.sin(t)
    return np.stack([dx, dy], axis=-1)

def normalize(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norms + 1e-12)

# ==============================================================================
# 2D Frame computation
# ==============================================================================

def get_2d_basis(t_vals, mode):
    d1 = curve_d1(t_vals)
    T = normalize(d1)
    
    # Standard 2D Normal (90 deg CCW rotation of Tangent)
    # n_standard = [-Ty, Tx]
    N_standard = np.stack([-T[..., 1], T[..., 0]], axis=-1)
    
    if mode == "RMF":
        # Bishop frame in 2D: The normal is simply the rotated tangent.
        # It never flips because it doesn't depend on the second derivative.
        return T, N_standard
    else:
        # Frenet-Serret in 2D: The normal is defined by the curvature.
        # N = d2_perp. This will snap 180 degrees at inflection points.
        d2 = curve_d2(t_vals)
        # Project d2 to be orthogonal to T
        d2_perp = d2 - np.sum(d2 * T, axis=-1, keepdims=True) * T
        N_fs = normalize(d2_perp) 
        return T, N_fs

# ==============================================================================
# Drawing helpers
# ==============================================================================

def draw_corridor(ax, p, e1, t_vals, alpha=0.2, color=C_CORRIDOR):
    # e1 is our normal vector. 
    # The boundaries are p + left * e1 and p - right * e1
    left_bound  = p + CORRIDOR_WIDTH_LEFT * e1
    right_bound = p - CORRIDOR_WIDTH_RIGHT * e1

    # Concatenate the points to form a single closed loop for the polygon
    # Left bound forward, then Right bound backward
    poly_pts = np.concatenate([left_bound, right_bound[::-1]], axis=0)
    
    # Draw the filled area
    ax.fill(poly_pts[:, 0], poly_pts[:, 1], color=color, alpha=alpha, lw=0)

    # Draw the boundary lines for definition
    ax.plot(left_bound[:, 0], left_bound[:, 1], color=color, alpha=0.6, lw=1.5)
    ax.plot(right_bound[:, 0], right_bound[:, 1], color=color, alpha=0.6, lw=1.5)

    # Highlight the "snap" in FS mode near the inflection point (t=pi)
    if FRAME_MODE == "FS":
        # Find indices immediately surrounding the inflection point
        before_inf = np.where(t_vals < np.pi)[0][-1]
        after_inf  = np.where(t_vals > np.pi)[0][0]
        
        for i in [before_inf, after_inf]:
            ax.plot([left_bound[i,0], right_bound[i,0]], 
                    [left_bound[i,1], right_bound[i,1]], 
                    color="black", alpha=0.4, lw=1.2, linestyle='--')

def draw_frames(ax, p, T, e1, length=FRAME_LENGTH):
    width=0.008
    for i in range(len(p)):
        # Tangent
        ax.quiver(p[i,0], p[i,1], T[i,0], T[i,1], color=C_T, 
                  scale=1/length, scale_units='xy', angles='xy', width=width)
        # Normal
        ax.quiver(p[i,0], p[i,1], e1[i,0], e1[i,1], color=C_E1, 
                  scale=1/length, scale_units='xy', angles='xy', width=width)

# ==============================================================================
# Main
# ==============================================================================

def main():
    t_start = np.pi - HALF_WINDOW
    t_end   = np.pi + HALF_WINDOW

    t = np.linspace(t_start, t_end, N_CURVE)
    pts = curve(t)

    t_sparse = np.linspace(t_start, t_end, N_FRAMES)
    p_sparse = curve(t_sparse)
    T_s, E1_s = get_2d_basis(t_sparse, FRAME_MODE)

    t_tube = np.linspace(t_start, t_end, N_TUBE)
    p_tube = curve(t_tube)
    _, E1_tube = get_2d_basis(t_tube, FRAME_MODE)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    
    # Inflection marker
    ax.scatter([np.pi], [0], s=100, color="darkorange", zorder=6, label="Inflection Point")

    # Centerline
    ax.plot(pts[:, 0], pts[:, 1], color=C_CURVE, lw=2, zorder=5)

    # Corridor
    draw_corridor(ax, p_tube, E1_tube, t_tube)

    # Sparse Frames
    draw_frames(ax, p_sparse, T_s, E1_s)

    # Polish
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    title = "Bishop Frame (2D): Consistent Corridor" if FRAME_MODE == "RMF" else \
            "Frenet-Serret Frame (2D): Discontinuous Corridor"
    # ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(pts[:,0].min() - 0.5, pts[:,0].max() + 0.5)
    ax.set_ylim(pts[:,1].min() - 0.5, pts[:,1].max() + 0.5)
    
    plt.tight_layout()
    print(f"Showing 2D {FRAME_MODE} visualization...")
    plt.show()

if __name__ == "__main__":
    main()
