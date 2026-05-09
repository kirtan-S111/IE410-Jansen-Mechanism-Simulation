"""
THEO JANSEN WALKING MECHANISM SIMULATION (CORRECTED)
IE410: Introduction to Robotics — Project Part B

Simulates the 8-bar Jansen-type gait-trainer linkage from
  Shin, Deshpande & Sulzer (2018), J. Mechanisms & Robotics 10(4), 044503
  Jadav et al., "Kinematic Performance of a Customizable Single DoF Gait Trainer"

Topology (Fig. 3 of Shin et al. / Fig. 1 of Jadav et al.):
  P0 : crank pivot (fixed, origin)
  P3 : ground pivot (fixed, P0 + L4 along +x)
  P1 : crank tip (rotates around P0 at radius L1)
  P2 : top vertex of upper four-bar  (|P1-P2|=L2, |P3-P2|=L3)
  P5 : bottom vertex of lower four-bar (|P1-P5|=L7, |P3-P5|=L8)
  P4 : RIGID-TRIANGLE vertex of coupler (P3,P2,P4) — sides L3,L5,L6
       (|P2-P4|=L5, |P3-P4|=L6)
  P6 : parallelogram corner (|P4-P6|=L10, |P5-P6|=L9)
  PE : foot tip — RIGID-TRIANGLE vertex of (P5,P6,PE) — sides L9,L11,L12
       (|P5-PE|=L11, |P6-PE|=L12)

The fix vs. the previous version: P4 is now solved as the third vertex of
the coupler triangle L3-L5-L6 (anchored on P2 and P3), NOT on P5 — and the
foot triangle uses the correct branch so PE traces the canonical D-shape
"shoe-sole" gait curve.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

# =============================================================================
# 1. NOMINAL LINK LENGTHS  (Table 3, Jadav et al., all in cm)
# =============================================================================
LINK_LENGTHS = {
    'L1':  11.0,   # crank          (* adjustable)
    'L2':  45.0,
    'L3':  36.0,
    'L4':  33.0,   # ground link    (* adjustable)
    'L5':  48.5,
    'L6':  41.5,
    'L7':  60.5,
    'L8':  41.5,   #                (* adjustable)
    'L9':  42.0,
    'L10': 43.0,
    'L11': 26.5,
    'L12': 54.5,
}

# =============================================================================
# 2. CIRCLE-CIRCLE INTERSECTION (vector-loop primitive)
# =============================================================================
def cci(C1, r1, C2, r2, branch='+'):
    """Intersection of two circles centred at C1, C2 with radii r1, r2.
    branch='+' returns the point on the LEFT of vector C1->C2,
    branch='-' returns the point on the RIGHT.
    Slight overshoot is clamped to keep the kinematics continuous.
    """
    x1, y1 = C1
    x2, y2 = C2
    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)
    # numerical clamp
    if d > r1 + r2:
        d = r1 + r2 - 1e-9
    elif d < abs(r1 - r2):
        d = abs(r1 - r2) + 1e-9
    a = (r1*r1 - r2*r2 + d*d) / (2.0 * d)
    h = np.sqrt(max(0.0, r1*r1 - a*a))
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    px, py = -dy / d, dx / d          # unit perpendicular (rotated +90°)
    s = 1.0 if branch == '+' else -1.0
    return (xm + s * h * px, ym + s * h * py)


# =============================================================================
# 3. FORWARD KINEMATICS — single crank angle -> all joint positions
# =============================================================================
def solve_jansen(theta2, L):
    P0 = (0.0, 0.0)
    P3 = (L['L4'], 0.0)

    # Crank tip
    P1 = (L['L1'] * np.cos(theta2),
          L['L1'] * np.sin(theta2))

    # Upper four-bar:  P2 lies above the line P1->P3
    P2 = cci(P1, L['L2'], P3, L['L3'], branch='+')

    # Lower four-bar:  P5 lies below the line P1->P3
    P5 = cci(P1, L['L7'], P3, L['L8'], branch='-')

    # Coupler RIGID TRIANGLE  (P2,P3,P4) with sides L3, L5, L6
    #   P4 lies on the OPPOSITE side of the line P2-P3 from P1
    #   (i.e. on the right of the vector P2->P3)
    P4 = cci(P2, L['L5'], P3, L['L6'], branch='-')

    # Parallelogram: P6 with |P4-P6|=L10, |P5-P6|=L9, on the right of P4->P5
    P6 = cci(P4, L['L10'], P5, L['L9'], branch='-')

    # Foot RIGID TRIANGLE  (P5,P6,PE) with sides L9, L11, L12
    #   PE hangs below the line P5-P6
    PE = cci(P5, L['L11'], P6, L['L12'], branch='-')

    return dict(P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, P6=P6, PE=PE)


def compute_trajectory(L, N=360):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    xs = np.zeros(N)
    ys = np.zeros(N)
    pos_list = []
    for i, th in enumerate(angles):
        try:
            p = solve_jansen(th, L)
            xs[i], ys[i] = p['PE']
            pos_list.append(p)
        except Exception:
            xs[i], ys[i] = np.nan, np.nan
            pos_list.append(None)
    return angles, xs, ys, pos_list


# =============================================================================
# 4. BASELINE TRAJECTORY
# =============================================================================
print("Computing baseline foot-point trajectory ...")
N = 360
angles, traj_x, traj_y, all_pos = compute_trajectory(LINK_LENGTHS, N)

stride = np.nanmax(traj_x) - np.nanmin(traj_x)
step_h = np.nanmax(traj_y) - np.nanmin(traj_y)
# Shoelace area
v = ~np.isnan(traj_x)
tx, ty = traj_x[v], traj_y[v]
area = 0.5 * abs(np.sum(tx[:-1]*ty[1:] - tx[1:]*ty[:-1])
                 + tx[-1]*ty[0] - tx[0]*ty[-1])
print(f"  Stride length (X span):   {stride:6.2f} cm")
print(f"  Step height   (Y span):   {step_h:6.2f} cm")
print(f"  Enclosed area:            {area:6.2f} cm²")


# =============================================================================
# 5. PLOT FOOT-POINT TRAJECTORY
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(traj_x, traj_y, 'b-', linewidth=2.5, label='Foot trajectory')
ax.plot(traj_x[0], traj_y[0], 'ro', markersize=10, zorder=5, label='Start (θ₂=0)')
ax.set_xlabel('X (cm)', fontsize=14)
ax.set_ylabel('Y (cm)', fontsize=14)
ax.set_title('Theo Jansen Mechanism — End-Effector (Foot) Trajectory',
             fontsize=15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig('foot_trajectory.png', dpi=150)
plt.close()
print("Saved: foot_trajectory.png")


# =============================================================================
# 6. DRAW THE FULL LINKAGE  (helper)
# =============================================================================
def draw_mechanism(ax, pos, trail_x=None, trail_y=None):
    P0, P1, P2, P3 = pos['P0'], pos['P1'], pos['P2'], pos['P3']
    P4, P5, P6, PE = pos['P4'], pos['P5'], pos['P6'], pos['PE']

    # Background trail
    if trail_x is not None and trail_y is not None and len(trail_x) > 1:
        ax.plot(trail_x, trail_y, '-', color='#cccccc', linewidth=1, zorder=1)

    links = [
        (P0, P3, '#444444', 3.0),  # L4 ground (drawn first)
        (P0, P1, '#cc2222', 3.0),  # L1 crank
        (P1, P2, '#2222cc', 2.4),  # L2
        (P3, P2, '#2222cc', 2.4),  # L3   (also a side of coupler triangle)
        (P2, P4, '#22aa22', 2.4),  # L5
        (P3, P4, '#22aa22', 2.4),  # L6
        (P1, P5, '#dd8800', 2.4),  # L7
        (P3, P5, '#dd8800', 2.4),  # L8
        (P4, P6, '#7733aa', 2.4),  # L10
        (P5, P6, '#7733aa', 2.4),  # L9   (also a side of foot triangle)
        (P5, PE, '#cc1188', 3.0),  # L11
        (P6, PE, '#cc1188', 3.0),  # L12
    ]
    for A, B, c, w in links:
        ax.plot([A[0], B[0]], [A[1], B[1]], '-', color=c, linewidth=w, zorder=2)

    # Joints
    for P, marker, sz, fc in [
        (P0, 's', 11, 'black'), (P3, 's', 11, 'black'),
        (P1, 'o', 7, '#cc2222'), (P2, 'o', 7, 'black'),
        (P4, 'o', 7, 'black'), (P5, 'o', 7, 'black'),
        (P6, 'o', 7, 'black'),
        (PE, 'o', 12, '#cc1188')]:
        ax.plot(P[0], P[1], marker, markersize=sz, markerfacecolor=fc,
                markeredgecolor='black', zorder=3)


# =============================================================================
# 7. SIX-POSITION SNAPSHOT PANEL
# =============================================================================
print("Generating mechanism snapshot panel ...")
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
snap_idx = [int(N * k / 6) for k in range(6)]

# global axis limits
all_x, all_y = [traj_x.copy().tolist()], [traj_y.copy().tolist()]
for p in all_pos:
    if p is None: continue
    for k in p:
        all_x.append([p[k][0]]); all_y.append([p[k][1]])
ax_xs = np.concatenate([np.array(a) for a in all_x])
ax_ys = np.concatenate([np.array(a) for a in all_y])
xlim = (ax_xs.min() - 8, ax_xs.max() + 8)
ylim = (ax_ys.min() - 8, ax_ys.max() + 8)

for k, idx in enumerate(snap_idx):
    ax = axs[k // 3, k % 3]
    p = all_pos[idx]
    if p is None: continue
    draw_mechanism(ax, p, traj_x, traj_y)
    ax.set_title(f'θ₂ = {np.degrees(angles[idx]):.0f}°', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
plt.suptitle('Mechanism Configuration at Six Crank Angles', fontsize=15)
plt.tight_layout()
plt.savefig('mechanism_snapshots.png', dpi=140)
plt.close()
print("Saved: mechanism_snapshots.png")


# =============================================================================
# 8. ANIMATION (GIF)
# =============================================================================
print("Generating animation frames (this may take a moment) ...")
n_frames = 60
frame_idx = np.linspace(0, N - 1, n_frames, dtype=int)
frames = []
tmpdir = '/tmp/jansen_frames'
os.makedirs(tmpdir, exist_ok=True)

for k, idx in enumerate(frame_idx):
    fig, ax = plt.subplots(figsize=(8, 7))
    p = all_pos[idx]
    if p is not None:
        draw_mechanism(ax, p, traj_x[:idx + 1], traj_y[:idx + 1])
        # red dot at current foot position
        ax.plot(traj_x[idx], traj_y[idx], 'o', color='red',
                markersize=10, zorder=4)
    ax.set_aspect('equal')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)')
    ax.set_title(f'Theo Jansen Mechanism   θ₂ = {np.degrees(angles[idx]):.0f}°',
                 fontsize=13)
    fp = f'{tmpdir}/frame_{k:03d}.png'
    plt.tight_layout()
    plt.savefig(fp, dpi=100)
    plt.close()
    frames.append(Image.open(fp))

frames[0].save('mechanism_animation.gif', save_all=True,
               append_images=frames[1:], duration=70, loop=0)
print("Saved: mechanism_animation.gif")


# =============================================================================
# 9. LINK-LENGTH VARIATION STUDY  (vary L4 and L8 — the adjustable links)
# =============================================================================
print("Studying link-length variation (L4, L8) ...")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Vary L4
for L4_val in [27, 30, 33, 36, 39]:
    L_var = LINK_LENGTHS.copy(); L_var['L4'] = L4_val
    _, x, y, _ = compute_trajectory(L_var, 240)
    axs[0].plot(x, y, '-', linewidth=2, label=f'L4={L4_val} cm')
axs[0].set_title('Effect of varying L4 (ground link)', fontsize=13)
axs[0].set_xlabel('X (cm)'); axs[0].set_ylabel('Y (cm)')
axs[0].set_aspect('equal'); axs[0].grid(alpha=0.3); axs[0].legend()

# Vary L8
for L8_val in [35.5, 38.5, 41.5, 44.5, 47.5]:
    L_var = LINK_LENGTHS.copy(); L_var['L8'] = L8_val
    _, x, y, _ = compute_trajectory(L_var, 240)
    axs[1].plot(x, y, '-', linewidth=2, label=f'L8={L8_val} cm')
axs[1].set_title('Effect of varying L8', fontsize=13)
axs[1].set_xlabel('X (cm)'); axs[1].set_ylabel('Y (cm)')
axs[1].set_aspect('equal'); axs[1].grid(alpha=0.3); axs[1].legend()

plt.tight_layout()
plt.savefig('link_length_variation.png', dpi=140)
plt.close()
print("Saved: link_length_variation.png")


# =============================================================================
# 10. COMPARISON WITH REFERENCE GAIT (meta-trajectory shape)
# =============================================================================
print("Building comparison with reference gait pattern ...")
# Approximate the meta-trajectory of Shin et al. Fig. 2:
#   stance: nearly flat (slight rise) bottom moving backward
#   swing : arched top moving forward
t_ref = np.linspace(0, 2 * np.pi, 200)
x_ref = 25.0 * np.cos(t_ref)
y_raw = 12.0 * np.sin(t_ref)
y_ref = np.where(y_raw > 0, y_raw, 0.6 * np.sin(2 * t_ref))

# Centre both for shape comparison
PE_xc = traj_x - np.nanmean(traj_x)
PE_yc = traj_y - np.nanmean(traj_y)
xr_c = x_ref - np.mean(x_ref)
yr_c = y_ref - np.mean(y_ref)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(xr_c, yr_c, 'k--', linewidth=2.5, label='Reference (meta-trajectory)')
ax.plot(PE_xc, PE_yc, 'b-', linewidth=2.5, label='Simulated Jansen mechanism')
ax.set_xlabel('X (cm)', fontsize=13)
ax.set_ylabel('Y (cm)', fontsize=13)
ax.set_title('Simulated Trajectory vs. Reference Human Gait Pattern',
             fontsize=14)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('gait_comparison.png', dpi=140)
plt.close()
print("Saved: gait_comparison.png")


# =============================================================================
# 11. TRAJECTORY VS. GAIT-CYCLE PERCENTAGE
# =============================================================================
gait_pct = np.linspace(0, 100, N)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.plot(gait_pct, traj_x, 'b-', linewidth=2)
ax1.set_ylabel('X (cm)', fontsize=12)
ax1.set_title('End-Effector Position over Gait Cycle', fontsize=14)
ax1.grid(alpha=0.3)
ax2.plot(gait_pct, traj_y, 'r-', linewidth=2)
ax2.set_xlabel('Gait Cycle (%)', fontsize=12)
ax2.set_ylabel('Y (cm)', fontsize=12)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('trajectory_vs_gait_cycle.png', dpi=140)
plt.close()
print("Saved: trajectory_vs_gait_cycle.png")

print("\n=== ALL OUTPUTS GENERATED ===")
print("  foot_trajectory.png")
print("  mechanism_snapshots.png")
print("  mechanism_animation.gif")
print("  link_length_variation.png")
print("  gait_comparison.png")
print("  trajectory_vs_gait_cycle.png")
