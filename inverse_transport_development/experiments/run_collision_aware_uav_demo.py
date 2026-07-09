# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-16
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-16
# 目的 / Purpose:
#   运行碰撞优先 UAV 反推链路的最小演示：从解析载荷轨迹生成 wrench 和参考
#   UAV 位置，再通过局部安全走廊 QP 做一次修正并输出张力可行性摘要。
#   Run a minimal demo of the collision-first UAV inference chain: generate a
#   payload trajectory, wrench, and reference UAV positions, then refine them
#   once with the local safe-corridor QP and report tension-feasibility stats.
# 主要输入 / Main Inputs:
#   内置解析轨迹、载荷参数、挂点几何与局部走廊配置。
#   Built-in analytic trajectory, payload parameters, attachment geometry, and
#   local corridor settings.
# 主要输出 / Main Outputs:
#   终端摘要，包含 QP 诊断和逐时刻张力可行性统计。
#   Console summary with QP diagnostics and per-sample tension-feasibility stats.
# ==============================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
	sys.path.insert(0, str(PKG_ROOT))

from src.cable_force_inference import PayloadGeometry, PayloadPhysicalParams, compute_payload_wrench, infer_cable_force_series
from src.common import build_three_uav_box_attachment_set
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
		source="collision_aware_demo",
	)


def build_rotation_series(trajectory: TrajectorySamples3D) -> np.ndarray:
	from src.common import rotation_matrix_from_rpy

	return np.asarray([rotation_matrix_from_rpy(rpy) for rpy in trajectory.orientation_rpy_rad], dtype=float)


def main() -> None:
	trajectory = build_demo_trajectory()
	attachments = build_three_uav_box_attachment_set(np.array([0.8, 0.3, 0.2], dtype=float))
	params = PayloadPhysicalParams(
		mass_kg=1.4,
		geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.array([0.8, 0.3, 0.2], dtype=float)),
	)
	wrench_series = compute_payload_wrench(trajectory, params)
	reference_series = infer_cable_force_series(trajectory, wrench_series, attachments)
	reference_positions_m = np.asarray(reference_series.quadrotor_position_inertial_m, dtype=float).copy()

	# Intentionally shrink the corridor center for one UAV at one sample so the QP
	# must actively project the reference back into the safe region.
	reference_positions_for_corridor = reference_positions_m.copy()
	mid_index = trajectory.time_s.size // 2
	reference_positions_for_corridor[mid_index, 0, 0] -= 0.10

	config = SequentialPlanningConfig(
		corridor_config=CorridorGenerationConfig(half_width_xyz_m=np.array([0.12, 0.22, 0.22], dtype=float)),
		kinematic_limits=UAVKinematicLimits(max_speed_mps=6.0, max_acceleration_mps2=12.0),
		tension_max_n=np.array([30.0, 30.0, 30.0], dtype=float),
		obstacles=(
			InflatedObstacleSphere(center_m=np.array([0.05, 0.20, 1.10], dtype=float), radius_m=0.12, label="demo_obs"),
		),
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
	)

	result = plan_collision_aware_uav_positions(
		time_s=trajectory.time_s,
		payload_positions_m=trajectory.position_m,
		payload_rotation_matrices=build_rotation_series(trajectory),
		reference_uav_positions_m=reference_positions_for_corridor,
		wrench_body_series=np.column_stack([wrench_series.force_n, wrench_series.torque_nm]),
		attachments=attachments,
		config=config,
	)

	residual_norms = np.array([item.residual_norm for item in result.tension_results], dtype=float)
	strict_count = int(sum(item.strictly_feasible for item in result.tension_results))
	relaxed_count = int(sum(item.relaxed_feasible for item in result.tension_results))
	force_rel = np.array([item.force_relative_residual for item in result.tension_results], dtype=float)
	torque_rel = np.array([item.torque_relative_residual for item in result.tension_results], dtype=float)
	print(f"[collision_demo] planner_status={result.status} qp_status={result.qp_result.status}")
	print("[collision_demo] obstacles=1 (inflated sphere tangent-halfspace corridor approximation)")
	print(
		"[collision_demo] "
		f"iterations_run={result.iterations_run}, "
		f"residual_history={np.array2string(result.residual_history, precision=3)}, "
		f"strict_history={np.array2string(result.strict_feasible_history)}, "
		f"relaxed_history={np.array2string(result.relaxed_feasible_history)}"
	)
	print(
		"[collision_demo] "
		f"corridor_max_violation={result.qp_result.corridor_max_violation_m:.6e} m, "
		f"separation_min_margin={result.qp_result.separation_min_margin_m:.6f} m"
	)
	print(
		"[collision_demo] "
		f"max_speed={result.qp_result.max_speed_mps:.3f} m/s, "
		f"max_acceleration={result.qp_result.max_acceleration_mps2:.3f} m/s^2"
	)
	print(
		"[collision_demo] "
		f"tension_strict={strict_count}/{len(result.tension_results)}, "
		f"tension_relaxed={relaxed_count}/{len(result.tension_results)}, "
		f"residual_norm[min/mean/max]={residual_norms.min():.3e}/{residual_norms.mean():.3e}/{residual_norms.max():.3e}"
	)
	print(
		"[collision_demo] "
		f"force_rel[min/mean/max]={force_rel.min():.3e}/{force_rel.mean():.3e}/{force_rel.max():.3e}, "
		f"torque_rel[min/mean/max]={torque_rel.min():.3e}/{torque_rel.mean():.3e}/{torque_rel.max():.3e}"
	)


if __name__ == "__main__":
	main()