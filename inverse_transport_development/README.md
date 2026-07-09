# Inverse Transport Development

This workspace is the independent development area for the next research stage of RL_Enhanced.

Its goal is to extend the current payload trajectory planning result toward:

1. inferring cable driving forces from planned payload trajectories,
2. inferring UAV positions and formation states from cable-force and payload motion constraints,
3. keeping new-stage research code, notes, references, and experiments separated from the current stage成果代码。

## Directory Structure

- `records/`: research records, stage summaries, and development notes.
- `references/`: papers, repository references, and evidence collection materials.
- `src/`: main source code for inverse modeling and supporting utilities.
- `experiments/`: temporary but structured experiment entry points and sanity studies.
- `data/`: raw inputs and processed intermediate artifacts for the new stage.
- `configs/`: experiment and solver configurations.
- `results/`: generated outputs, plots, logs, and derived artifacts.
- `tests/`: focused validation scripts and test cases.

## Source Layout

- `src/cable_force_inference/`: inverse force reconstruction from payload trajectory and motion constraints.
- `src/uav_pose_inference/`: UAV position and formation reconstruction from cable and payload states.
- `src/common/`: shared math, geometry, IO, and utility modules.

## Input Decision For Wrench Stage

For the first wrench stage, the canonical input should be a time-stamped payload trajectory rather than a velocity-only stream.

Recommended priority:

1. Use piecewise polynomial trajectory coefficients plus segment durations when available, because velocity and acceleration can then be recovered analytically.
2. Fall back to `sample_t + sample_xyz` when only sampled positions are available.
3. Treat explicit velocity and acceleration as optional overrides, mainly for external simulators or estimator outputs.

Current evidence from the existing planning pipeline:

- `fixed_scene_3d_demo_grid.py` exports `sample_t`, `sample_xyz`, `coeffs_x`, `coeffs_y`, `coeffs_z`, and `T_per_seg`.
- `traj_qp_corridor.py` currently samples the solved 3D trajectory with a default spacing near `dt=0.08 s`.

This means the current planner already outputs enough information to support a first-stage wrench solver without adding a ROS or Gazebo dependency.

## Torque-Stage Scope

The current second-stage wrench scaffold now distinguishes between:

1. translational force recovery from payload acceleration,
2. rigid-body torque recovery from angular velocity, angular acceleration, and inertia,
3. later attachment-point and cable-force modeling, which is intentionally deferred.

At this stage, payload geometry is only used to provide an inertia interface, with a uniform box inertia model as the first supported option.

The framework now also exposes the rigid-body attachment variables used in the classical model:

- `r_i`: attachment-point positions in the load body frame,
- `L_i`: cable lengths,
- `q_i`: cable directions in the load body frame,
- equation (5) kinematics for recovering quadrotor positions,
- a body-frame wrench map `Phi` such that `Phi T = W` for scalar cable tensions.

## Working Rules

- Keep temporary scripts out of the repository root.
- Record major conclusions and next actions in `records/总结.md`.
- Add external evidence into `references/` before adopting nontrivial models or assumptions.
- Keep core modules in `src/` and place one-off experiments in `experiments/`.