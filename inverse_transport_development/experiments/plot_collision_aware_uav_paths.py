# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-16
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-16
# 目的 / Purpose:
#   运行碰撞优先 UAV 反推 demo，并输出带障碍球的参考/规划后 UAV 轨迹对比图，
#   用于直接检查避障效果是否可见。
#   Run the collision-aware UAV inference demo and export obstacle-aware plots
#   comparing reference and refined UAV trajectories to directly inspect the
#   visibility of obstacle avoidance.
# 主要输入 / Main Inputs:
#   内置解析载荷轨迹、载荷参数、挂点几何、顺序规划配置与球形障碍。
#   Built-in analytic payload trajectory, payload parameters, attachment
#   geometry, sequential-planning configuration, and a spherical obstacle.
# 主要输出 / Main Outputs:
#   参考/规划后 UAV 轨迹对比图、障碍投影图和 npz 结果包。
#   Comparison plots for reference/refined UAV paths, obstacle projection plots,
#   and an npz result bundle.
# ==============================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
	sys.path.insert(0, str(PKG_ROOT))

from src.cable_force_inference import PayloadGeometry, PayloadPhysicalParams, compute_payload_wrench, infer_cable_force_series
from src.common import build_three_uav_box_attachment_set, rotation_matrix_from_rpy
from src.common.trajectory import TrajectorySamples3D
from src.uav_pose_inference import (
	CorridorGenerationConfig,
	InflatedObstacleSphere,
	SequentialPlanningConfig,
	UAVKinematicLimits,
	plan_collision_aware_uav_positions,
)


def build_demo_trajectory() -> TrajectorySamples3D:
	time_s = np.linspace(0.0, 2.0, 41)
	position_m = np.column_stack(
		[
			0.25 * np.sin(2.0 * np.pi * time_s / 2.0),
			0.1 * np.cos(2.0 * np.pi * time_s / 2.0),
			0.7 + 0.05 * np.sin(4.0 * np.pi * time_s / 2.0),
		]
	)
	orientation_rpy_rad = np.column_stack(
		[
			0.03 * np.sin(2.0 * np.pi * time_s / 2.0),
			0.02 * np.cos(2.0 * np.pi * time_s / 2.0),
			0.04 * np.sin(2.0 * np.pi * time_s / 2.0),
		]
	)
	return TrajectorySamples3D.from_position_samples(
		time_s=time_s,
		position_m=position_m,
		orientation_rpy_rad=orientation_rpy_rad,
		source="collision_aware_plot_demo",
	)


def build_rotation_series(trajectory: TrajectorySamples3D) -> np.ndarray:
	return np.asarray([rotation_matrix_from_rpy(rpy) for rpy in trajectory.orientation_rpy_rad], dtype=float)


def compute_min_clearance_to_sphere(positions_m: np.ndarray, obstacle: InflatedObstacleSphere) -> float:
	positions_m = np.asarray(positions_m, dtype=float)
	distances = np.linalg.norm(positions_m - obstacle.center_m[None, None, :], axis=2)
	return float(np.min(distances - obstacle.radius_m))


def add_circle(axis, center_x: float, center_y: float, radius: float, color: str, label: str) -> None:
	circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.18, label=label)
	axis.add_patch(circle)


def save_xy_plot(output_path: Path, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, payload_positions_m: np.ndarray, obstacle: InflatedObstacleSphere) -> None:
	fig, axis = plt.subplots(figsize=(8, 7))
	add_circle(axis, float(obstacle.center_m[0]), float(obstacle.center_m[1]), float(obstacle.radius_m), "tab:red", f"obstacle r={obstacle.radius_m:.2f}")
	axis.plot(payload_positions_m[:, 0], payload_positions_m[:, 1], color="black", linewidth=2.0, label="payload")
	for uav_index in range(reference_positions_m.shape[1]):
		axis.plot(reference_positions_m[:, uav_index, 0], reference_positions_m[:, uav_index, 1], linestyle="--", alpha=0.7, label=f"ref_uav{uav_index+1}")
		axis.plot(refined_positions_m[:, uav_index, 0], refined_positions_m[:, uav_index, 1], linewidth=2.0, label=f"plan_uav{uav_index+1}")
	axis.set_xlabel("x [m]")
	axis.set_ylabel("y [m]")
	axis.set_title("XY Projection With Obstacle")
	axis.grid(True, alpha=0.3)
	axis.axis("equal")
	axis.legend(loc="best", ncol=2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def save_xz_plot(output_path: Path, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, payload_positions_m: np.ndarray, obstacle: InflatedObstacleSphere) -> None:
	fig, axis = plt.subplots(figsize=(8, 7))
	add_circle(axis, float(obstacle.center_m[0]), float(obstacle.center_m[2]), float(obstacle.radius_m), "tab:red", f"obstacle r={obstacle.radius_m:.2f}")
	axis.plot(payload_positions_m[:, 0], payload_positions_m[:, 2], color="black", linewidth=2.0, label="payload")
	for uav_index in range(reference_positions_m.shape[1]):
		axis.plot(reference_positions_m[:, uav_index, 0], reference_positions_m[:, uav_index, 2], linestyle="--", alpha=0.7, label=f"ref_uav{uav_index+1}")
		axis.plot(refined_positions_m[:, uav_index, 0], refined_positions_m[:, uav_index, 2], linewidth=2.0, label=f"plan_uav{uav_index+1}")
	axis.set_xlabel("x [m]")
	axis.set_ylabel("z [m]")
	axis.set_title("XZ Projection With Obstacle")
	axis.grid(True, alpha=0.3)
	axis.axis("equal")
	axis.legend(loc="best", ncol=2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def save_3d_plot(output_path: Path, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, payload_positions_m: np.ndarray, obstacle: InflatedObstacleSphere) -> None:
	fig = plt.figure(figsize=(9, 8))
	axis = fig.add_subplot(111, projection="3d")
	u = np.linspace(0.0, 2.0 * np.pi, 48)
	v = np.linspace(0.0, np.pi, 24)
	x = obstacle.center_m[0] + obstacle.radius_m * np.outer(np.cos(u), np.sin(v))
	y = obstacle.center_m[1] + obstacle.radius_m * np.outer(np.sin(u), np.sin(v))
	z = obstacle.center_m[2] + obstacle.radius_m * np.outer(np.ones_like(u), np.cos(v))
	axis.plot_surface(x, y, z, color="tab:red", alpha=0.18, linewidth=0.0)
	axis.plot(payload_positions_m[:, 0], payload_positions_m[:, 1], payload_positions_m[:, 2], color="black", linewidth=2.0, label="payload")
	for uav_index in range(reference_positions_m.shape[1]):
		axis.plot(reference_positions_m[:, uav_index, 0], reference_positions_m[:, uav_index, 1], reference_positions_m[:, uav_index, 2], linestyle="--", alpha=0.7, label=f"ref_uav{uav_index+1}")
		axis.plot(refined_positions_m[:, uav_index, 0], refined_positions_m[:, uav_index, 1], refined_positions_m[:, uav_index, 2], linewidth=2.0, label=f"plan_uav{uav_index+1}")
	axis.set_xlabel("x [m]")
	axis.set_ylabel("y [m]")
	axis.set_zlabel("z [m]")
	axis.set_title("3D UAV Paths With Obstacle Sphere")
	axis.legend(loc="best", ncol=2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot obstacle-aware reference/refined UAV trajectories for the collision-aware demo")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("inverse_transport_development/results/collision_aware_demo_plots"),
		help="Directory used to save obstacle-aware comparison plots",
	)
	parser.add_argument("--obstacle-center", type=float, nargs=3, default=[0.05, 0.20, 1.10], help="Obstacle sphere center xyz in meters")
	parser.add_argument("--obstacle-radius", type=float, default=0.12, help="Obstacle sphere radius in meters")
	args = parser.parse_args()

	trajectory = build_demo_trajectory()
	attachments = build_three_uav_box_attachment_set(np.array([0.8, 0.3, 0.2], dtype=float))
	params = PayloadPhysicalParams(
		mass_kg=1.4,
		geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.array([0.8, 0.3, 0.2], dtype=float)),
	)
	wrench_series = compute_payload_wrench(trajectory, params)
	reference_series = infer_cable_force_series(trajectory, wrench_series, attachments)
	reference_positions_m = np.asarray(reference_series.quadrotor_position_inertial_m, dtype=float).copy()
	obstacle = InflatedObstacleSphere(center_m=np.asarray(args.obstacle_center, dtype=float), radius_m=float(args.obstacle_radius), label="demo_obs")
	result = plan_collision_aware_uav_positions(
		time_s=trajectory.time_s,
		payload_positions_m=trajectory.position_m,
		payload_rotation_matrices=build_rotation_series(trajectory),
		reference_uav_positions_m=reference_positions_m,
		wrench_body_series=np.column_stack([wrench_series.force_n, wrench_series.torque_nm]),
		attachments=attachments,
		config=SequentialPlanningConfig(
			corridor_config=CorridorGenerationConfig(half_width_xyz_m=np.array([0.12, 0.22, 0.22], dtype=float)),
			kinematic_limits=UAVKinematicLimits(max_speed_mps=6.0, max_acceleration_mps2=12.0),
			tension_max_n=np.array([30.0, 30.0, 30.0], dtype=float),
			obstacles=(obstacle,),
			max_iterations=4,
			length_tolerance_m=0.02,
			force_residual_absolute_tolerance_n=0.5,
			force_residual_relative_tolerance=0.05,
			torque_residual_absolute_tolerance_nm=0.12,
			torque_residual_relative_tolerance=2.50,
			force_relative_scale_floor_n=1.0,
			torque_relative_scale_floor_nm=0.05,
			local_reference_weight=1.0,
			tension_surrogate_weight=0.25,
			tension_linearization_weight=0.50,
			local_shape_weight=0.05,
			local_smoothness_weight=0.05,
			min_separation_m=0.10,
			trust_region_half_width_m=0.05,
			tension_linearization_delta_m=1e-3,
			feedback_force_gain_m_per_n=0.0,
			feedback_torque_gain_m_per_nm=0.10,
			feedback_max_step_m=0.04,
		),
	)
	refined_positions_m = np.asarray(result.positions_m, dtype=float)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	save_xy_plot(args.output_dir / "collision_aware_uav_xy.png", reference_positions_m, refined_positions_m, trajectory.position_m, obstacle)
	save_xz_plot(args.output_dir / "collision_aware_uav_xz.png", reference_positions_m, refined_positions_m, trajectory.position_m, obstacle)
	save_3d_plot(args.output_dir / "collision_aware_uav_3d.png", reference_positions_m, refined_positions_m, trajectory.position_m, obstacle)
	np.savez_compressed(
		args.output_dir / "collision_aware_uav_paths.npz",
		time_s=trajectory.time_s,
		payload_position_m=trajectory.position_m,
		reference_uav_positions_m=reference_positions_m,
		refined_uav_positions_m=refined_positions_m,
		obstacle_center_m=obstacle.center_m,
		obstacle_radius_m=obstacle.radius_m,
		residual_history=result.residual_history,
		status=result.status,
	)
	reference_clearance_m = compute_min_clearance_to_sphere(reference_positions_m, obstacle)
	refined_clearance_m = compute_min_clearance_to_sphere(refined_positions_m, obstacle)
	print(f"[collision_aware_plot] planner_status={result.status}")
	print(f"[collision_aware_plot] residual_history={np.array2string(result.residual_history, precision=3)}")
	print(f"[collision_aware_plot] obstacle_center={obstacle.center_m.tolist()} radius={obstacle.radius_m:.3f}")
	print(f"[collision_aware_plot] reference_min_clearance={reference_clearance_m:.6f} m")
	print(f"[collision_aware_plot] refined_min_clearance={refined_clearance_m:.6f} m")
	print(f"[collision_aware_plot] saved {args.output_dir / 'collision_aware_uav_xy.png'}")
	print(f"[collision_aware_plot] saved {args.output_dir / 'collision_aware_uav_xz.png'}")
	print(f"[collision_aware_plot] saved {args.output_dir / 'collision_aware_uav_3d.png'}")
	print(f"[collision_aware_plot] saved {args.output_dir / 'collision_aware_uav_paths.npz'}")


if __name__ == "__main__":
	main()