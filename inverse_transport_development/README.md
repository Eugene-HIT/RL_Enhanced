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

## Working Rules

- Keep temporary scripts out of the repository root.
- Record major conclusions and next actions in `records/总结.md`.
- Add external evidence into `references/` before adopting nontrivial models or assumptions.
- Keep core modules in `src/` and place one-off experiments in `experiments/`.