# IE410 Project Part B: Theo Jansen Walking Mechanism Simulation

## Overview
This repository contains the simulation, code, and report for **Part B: Simulation of a Theo Jansen Walking Mechanism for Gait Generation**, completed for the **IE410: Introduction to Robotics** course (Winter 2026).

The objective of this project is to model and simulate a single degree-of-freedom (DoF) linkage mechanism driven by a single actuator to produce a natural, complex gait-like end-effector trajectory.

## Team: Group 21
* **Kirtan Chaudhari** (202401095)
* **Hingrajiya Khush** (202401068)
* **Dharmesh Upadhyay** (202401049)
* **Tasvi Bhalani** (202401027)
* **Shyam Ramani** (202401175)

## Repository Structure
* `/code` - Contains the simulation models (MATLAB/Simulink/Simscape or MuJoCo).
* `/report` - Contains the LaTeX source code and the final project PDF report.
* `/images` - Contains trajectory plots and linkage diagrams extracted from the simulation.
* `/media` - Contains the animation/video demonstrating the mechanism in motion over a complete cycle.

## Project Tasks Completed
- [x] Modeled the 12-link Theo Jansen mechanism.
- [x] Simulated at least one complete motion cycle driven by a single rotary input.
- [x] Extracted and plotted the endpoint ("foot") trajectory in the x-y plane.
- [x] Conducted parametric sweeps to study the effect of linkage geometry (e.g., Lower Rocker L8, Crank L1) on the gait shape.
- [x] Compared the generated curve with natural walking trajectories from academic literature.

## How to Run the Simulation
1. Open the `/code` directory.
2. Load the main simulation file in [Specify your software here: MATLAB/MuJoCo].
3. Run the script to generate the linkage animation and plot the endpoint trajectory.

## Preview
![Gait Trajectory Animation](media/trajectory.gif)
*(Above: A preview of the simulated ankle trajectory during a full crank rotation).*
