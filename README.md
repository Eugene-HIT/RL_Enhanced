# RL_Enhanced

RL_Enhanced is a research-oriented prototype for learning payload passing configurations with reinforcement learning and then generating smooth corridor-constrained trajectories. The project uses a SAC policy on a grid-encoded door geometry environment to predict feasible passing poses, and combines the resulting keyframes with minimum-snap trajectory generation for downstream visualization and export.

The repository currently includes:

- SAC training for the passing-configuration policy.
- Environment and geometry encoding for door-shaped openings.
- Evaluation scripts for success rate, inference time, and reward statistics.
- MATLAB/Python visualization utilities for corridor export, trajectory derivatives, and 3D scene rendering.
- A demo video of the current result: [DemoVideo.mp4](DemoVideo.mp4).

## Main entry points

- [train.py](train.py): train the SAC passing-configuration policy.
- [env_passenv_grid.py](env_passenv_grid.py): RL environment with grid-based geometry observations.
- [fixed_scene_3d_demo_grid.py](fixed_scene_3d_demo_grid.py): generate a fixed-scene demo and export artifacts.
- [evaluate_model_stats.py](evaluate_model_stats.py): evaluate policy performance over sampled scenarios.
- [plot_minsnap_advantage.m](plot_minsnap_advantage.m): compare minimum-snap trajectories against simpler baselines.

## Notes

This codebase is still organized like an active research workspace rather than a polished package. The main training, evaluation, export, and visualization paths are kept in the repository; temporary root-level test scripts have been removed.
