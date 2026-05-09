import mujoco
import mujoco.viewer
import time
import numpy as np

def main():
    xml_path = "../Theo Jansen mechanism.xml"
    print(f"Loading {xml_path}...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Opening MuJoCo viewer. Close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Check if there is an actuator to control
        if model.nu > 0:
            # Set a constant velocity/effort for the first actuator
            # (adjust value as needed depending on if it's position, velocity, or motor)
            data.ctrl[0] = np.pi 
        
        # Settle for a few steps
        for _ in range(50):
            mujoco.mj_step(model, data)

        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # Try to run at roughly real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
