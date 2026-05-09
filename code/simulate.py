import mujoco
import numpy as np
import matplotlib.pyplot as plt
import imageio

def run_simulation(model_path="jansen.xml", save_gif=True, save_plot=True):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if save_gif:
        renderer = mujoco.Renderer(model, 400, 600)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.lookat = [0.15, 0, -0.2]
        camera.distance = 1.5
        camera.azimuth = 90
        camera.elevation = 0
        frames = []

    foot_site_id = model.site('s_foot').id
    
    # Validation data
    traj_x = []
    traj_z = []
    max_violation = 0.0

    site_pairs = [
        (model.site('s_P2_link2').id, model.site('s_P2_tri').id),
        (model.site('s_P5_link7').id, model.site('s_P5_link8').id),
        (model.site('s_P6_link10').id, model.site('s_P6_foot').id)
    ]
    
    def check_violation(max_v):
        v = 0.0
        for s1, s2 in site_pairs:
            dist = np.linalg.norm(data.site_xpos[s1] - data.site_xpos[s2])
            if dist > v: v = dist
        return max(max_v, v)

    # 1. Settle for 50 steps
    data.ctrl[0] = 0.0
    for _ in range(50):
        mujoco.mj_step(model, data)
        max_violation = check_violation(max_violation)

    # 2. Drive from 0 to 4*pi over 4 seconds at 1000 Hz
    steps = 4000
    for i in range(steps + 1):
        t = i * 0.001
        data.ctrl[0] = np.pi * t # 0 to 4*pi over 4 seconds
        mujoco.mj_step(model, data)
        
        pos = data.site_xpos[foot_site_id]
        traj_x.append(pos[0])
        traj_z.append(pos[2])
        
        if model.neq > 0:
            max_violation = check_violation(max_violation)
                
        if save_gif and i % 20 == 0:
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels)
            
    if save_gif:
        imageio.mimsave("mechanism.gif", frames, fps=50, loop=0)
        
    stride = max(traj_x) - min(traj_x)
    step_h = max(traj_z) - min(traj_z)
    
    p_start = np.array([traj_x[2000], traj_z[2000]])
    p_end = np.array([traj_x[4000], traj_z[4000]])
    closure_err = np.linalg.norm(p_start - p_end)
    
    print("=== VALIDATION ===")
    print(f"Stride length (max X - min X): {stride:.4f} m (should be 0.40 to 0.50)")
    print(f"Step height   (max Z - min Z): {step_h:.4f} m (should be 0.20 to 0.26)")
    print(f"Constraint violation max(|eq_err|): {max_violation:.6f} m (< 1e-3)")
    print(f"PE traces a CLOSED loop error: {closure_err*1000:.2f} mm (< 5 mm)")
    
    if save_plot:
        plt.figure(figsize=(8,6))
        # Plot only the second revolution for a clean loop
        plt.plot(traj_x[2000:], traj_z[2000:], 'b-', linewidth=2)
        plt.title('Theo Jansen Foot Trajectory (Sagittal Plane)')
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('foot_trajectory.png')

if __name__ == "__main__":
    run_simulation()
