# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-16
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-16
# 目的 / Purpose:
#   对碰撞优先 UAV 反推骨架做最小可执行验证，覆盖局部走廊 QP 和顺序规划
#   主循环的关键接口行为。
#   Provide minimal executable validation for the collision-first UAV inference
#   scaffold, covering the local corridor QP and the sequential planning loop.
# 主要输入 / Main Inputs:
#   内置合成轨迹、挂点几何、局部走廊与张力上界。
#   Built-in synthetic trajectories, attachment geometry, local corridors, and
#   tension upper bounds.
# 主要输出 / Main Outputs:
#   unittest 断言结果。
#   unittest assertion results.
# ==============================================

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
	LocalQPProblem,
	SequentialPlanningConfig,
	UAVKinematicLimits,
	build_axis_aligned_corridors_from_reference,
	plan_collision_aware_uav_positions,
	solve_local_corridor_qp,
)


def build_demo_trajectory() -> TrajectorySamples3D:
	time_s = np.linspace(0.0, 1.0, 5)
	position_m = np.column_stack(
		[
			0.1 * time_s,
			0.0 * time_s,
			0.8 + 0.02 * np.sin(2.0 * np.pi * time_s),
		]
	)
	orientation_rpy_rad = np.column_stack(
		[
			0.01 * np.sin(2.0 * np.pi * time_s),
			0.0 * time_s,
			0.01 * np.cos(2.0 * np.pi * time_s),
		]
	)
	return TrajectorySamples3D.from_position_samples(
		time_s=time_s,
		position_m=position_m,
		orientation_rpy_rad=orientation_rpy_rad,
		source="uav_pose_test_demo",
	)


class UAVPoseInferenceTests(unittest.TestCase):
	def test_obstacle_halfspace_shrinks_corridor_away_from_sphere(self) -> None:
		reference_positions_m = np.array([[[1.0, 0.0, 0.0]]], dtype=float)
		obstacle = InflatedObstacleSphere(center_m=np.array([0.0, 0.0, 0.0], dtype=float), radius_m=0.6, label="obs")
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=reference_positions_m,
			config=CorridorGenerationConfig(np.array([0.5, 0.5, 0.5], dtype=float)),
			obstacles=[obstacle],
		)
		corridor = corridors[0][0]
		self.assertTrue(corridor.contains(np.array([1.0, 0.0, 0.0], dtype=float)))
		self.assertFalse(corridor.contains(np.array([0.0, 0.0, 0.0], dtype=float)))
		self.assertGreaterEqual(corridor.A.shape[0], 7)

	def test_local_qp_projects_reference_back_into_corridor(self) -> None:
		time_s = np.array([0.0, 0.5, 1.0], dtype=float)
		reference_positions_m = np.array(
			[
				[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
				[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
				[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
			],
			dtype=float,
		)
		corridor_reference = reference_positions_m.copy()
		corridor_reference[1, 0, 0] = 0.2
		corridor_reference[1, 1, 0] = 1.2
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=corridor_reference,
			config=CorridorGenerationConfig(np.array([0.05, 0.2, 0.2], dtype=float)),
		)
		result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=reference_positions_m,
				corridors=corridors,
				kinematic_limits=UAVKinematicLimits(10.0, 20.0),
				reference_weight=1.0,
				smoothness_weight=0.1,
				min_separation_m=0.5,
			)
		)
		self.assertLessEqual(result.corridor_max_violation_m, 2e-6)
		self.assertGreaterEqual(result.separation_min_margin_m, -1e-6)
		self.assertLess(abs(result.positions_m[1, 0, 0] - 0.2), 0.06)
		self.assertLess(abs(result.positions_m[1, 1, 0] - 1.2), 0.06)

	def test_local_qp_tension_surrogate_uses_sample_weights(self) -> None:
		time_s = np.array([0.0, 1.0], dtype=float)
		reference_positions_m = np.zeros((2, 1, 3), dtype=float)
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=reference_positions_m,
			config=CorridorGenerationConfig(np.array([5.0, 5.0, 5.0], dtype=float)),
		)
		tension_surrogate_positions_m = np.array(
			[
				[[1.0, 0.0, 0.0]],
				[[1.0, 0.0, 0.0]],
			],
			dtype=float,
		)
		result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=reference_positions_m,
				corridors=corridors,
				kinematic_limits=UAVKinematicLimits(10.0, 20.0),
				reference_weight=1.0,
				tension_surrogate_positions_m=tension_surrogate_positions_m,
				tension_surrogate_sample_weights=np.array([1.0, 0.0], dtype=float),
				tension_surrogate_weight=1.0,
			)
		)
		self.assertAlmostEqual(result.positions_m[0, 0, 0], 0.5, delta=1e-3)
		self.assertAlmostEqual(result.positions_m[1, 0, 0], 0.0, delta=1e-3)

	def test_local_qp_shape_regularizer_preserves_pairwise_offset(self) -> None:
		time_s = np.array([0.0], dtype=float)
		reference_positions_m = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=float)
		corridor_reference = reference_positions_m.copy()
		corridor_reference[0, 0, 0] = 0.2
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=corridor_reference,
			config=CorridorGenerationConfig(np.array([0.01, 0.5, 0.5], dtype=float)),
		)
		baseline_result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=reference_positions_m,
				corridors=corridors,
				kinematic_limits=UAVKinematicLimits(10.0, 20.0),
				reference_weight=0.01,
				shape_weight=0.0,
			)
		)
		shape_result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=reference_positions_m,
				corridors=corridors,
				kinematic_limits=UAVKinematicLimits(10.0, 20.0),
				reference_weight=0.01,
				shape_weight=1.0,
			)
		)
		self.assertAlmostEqual(shape_result.positions_m[0, 0, 0], 0.19, delta=0.02)
		self.assertGreater(shape_result.positions_m[0, 1, 0], baseline_result.positions_m[0, 1, 0] + 1e-4)

	def test_local_qp_linearized_tension_term_respects_trust_region(self) -> None:
		time_s = np.array([0.0], dtype=float)
		reference_positions_m = np.zeros((1, 1, 3), dtype=float)
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=reference_positions_m,
			config=CorridorGenerationConfig(np.array([5.0, 5.0, 5.0], dtype=float)),
		)
		result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=reference_positions_m,
				corridors=corridors,
				kinematic_limits=UAVKinematicLimits(10.0, 20.0),
				reference_weight=1e-9,
				tension_linearized_residuals=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
				tension_linearized_jacobians=np.array([[[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]], dtype=float),
				tension_linearization_weight=1.0,
				trust_region_half_width_m=0.1,
			)
		)
		self.assertAlmostEqual(result.positions_m[0, 0, 0], 0.1, delta=5e-6)

	def test_sequential_planner_returns_tension_diagnostics(self) -> None:
		trajectory = build_demo_trajectory()
		attachments = build_three_uav_box_attachment_set(np.array([0.8, 0.3, 0.2], dtype=float))
		params = PayloadPhysicalParams(
			mass_kg=1.2,
			geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.array([0.8, 0.3, 0.2], dtype=float)),
		)
		wrench = compute_payload_wrench(trajectory, params)
		reference_series = infer_cable_force_series(trajectory, wrench, attachments)
		rotation_series = np.asarray([rotation_matrix_from_rpy(rpy) for rpy in trajectory.orientation_rpy_rad], dtype=float)
		result = plan_collision_aware_uav_positions(
			time_s=trajectory.time_s,
			payload_positions_m=trajectory.position_m,
			payload_rotation_matrices=rotation_series,
			reference_uav_positions_m=reference_series.quadrotor_position_inertial_m,
			wrench_body_series=np.column_stack([wrench.force_n, wrench.torque_nm]),
			attachments=attachments,
			config=SequentialPlanningConfig(
				corridor_config=CorridorGenerationConfig(np.array([0.12, 0.12, 0.12], dtype=float)),
				kinematic_limits=UAVKinematicLimits(5.0, 8.0),
				tension_max_n=np.array([30.0, 30.0, 30.0], dtype=float),
				obstacles=(InflatedObstacleSphere(center_m=np.array([2.0, 2.0, 2.0], dtype=float), radius_m=0.1),),
				max_iterations=3,
				local_reference_weight=1.0,
				tension_linearization_weight=0.50,
				local_shape_weight=0.05,
				local_smoothness_weight=0.05,
				min_separation_m=0.1,
				trust_region_half_width_m=0.05,
				feedback_force_gain_m_per_n=0.0,
				feedback_torque_gain_m_per_nm=0.1,
			),
		)
		self.assertEqual(result.positions_m.shape, reference_series.quadrotor_position_inertial_m.shape)
		self.assertEqual(len(result.tension_results), trajectory.time_s.size)
		self.assertLessEqual(result.qp_result.corridor_max_violation_m, 2e-6)
		self.assertTrue(np.isfinite(result.qp_result.max_speed_mps))
		self.assertTrue(np.isfinite(result.qp_result.max_acceleration_mps2))
		self.assertTrue(all(np.isfinite(item.force_relative_residual) for item in result.tension_results))
		self.assertTrue(all(np.isfinite(item.torque_relative_residual) for item in result.tension_results))
		self.assertIn(result.status, {"reference_strictly_feasible", "reference_relaxed_feasible", "reference_infeasible"})
		self.assertGreaterEqual(result.iterations_run, 1)
		self.assertEqual(result.residual_history.shape[0], result.iterations_run)
		self.assertTrue(np.all(np.isfinite(result.residual_history)))
		self.assertLessEqual(result.mean_residual_norm, np.min(result.residual_history) + 1e-9)


if __name__ == "__main__":
	unittest.main()