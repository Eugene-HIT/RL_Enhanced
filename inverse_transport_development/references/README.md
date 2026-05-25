# References

This folder stores the evidence base for the next-stage research and development of inverse transport modeling.

Current objective:

1. infer cable driving forces from planned payload trajectories,
2. infer UAV positions and formation states from cable-force and payload geometry constraints,
3. keep paper evidence and implementation references separated from core source code.

## Structure

- `papers/`: literature notes, curated reading lists, and paper-to-task mappings.
- `repos/`: open-source repository notes and implementation value assessments.

## Current Notes

- `papers/基础调研.md`: first-pass literature screening for inverse force and UAV pose inference.
- `repos/基础调研.md`: first-pass repository screening for simulation and validation baselines.

## Evidence Basis

The current notes are based on:

1. the local project paper [Reinforcement_Learning_Based_Passing_Configuration_Planning_for_Narrow_Opening_Slung_Load_Transportation.pdf](../../Reinforcement_Learning_Based_Passing_Configuration_Planning_for_Narrow_Opening_Slung_Load_Transportation.pdf), especially its abstract and references section,
2. GitHub repository README pages for RotorTM, RotorS, gym-pybullet-drones, and Crazyswarm2,
3. direct code observations from the current RL_Enhanced repository.

## Usage Rule

- Add only sources that are likely to change modeling choices, solver design, validation strategy, or system architecture.
- For each new source, record what is confirmed, what is inferred, and what still needs verification.