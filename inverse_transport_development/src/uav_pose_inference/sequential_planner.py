# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-10
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-10
# 目的 / Purpose:
#   定义碰撞优先 UAV 反推的顺序规划主循环骨架，并将当前参考轨迹、走廊构造、
#   局部 QP 接口和张力可行性检查串起来。
#   Define the sequential planning scaffold for collision-first UAV inference
#   and connect the current reference trajectory, corridor generation, local QP
#   interface, and inner tension-feasibility checks.
# 主要输入 / Main Inputs:
#   时间戳、载荷位姿、参考 UAV 轨迹、wrench 序列、挂点几何与系统约束。
#   Time stamps, payload pose, reference UAV trajectories, wrench sequence,
#   attachment geometry, and system limits.
# 主要输出 / Main Outputs:
#   第一版顺序规划结果，包括轨迹、走廊诊断与逐时刻张力可行性结果。
#   First-pass sequential planning results, including trajectories, corridor
#   diagnostics, and per-sample tension-feasibility results.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
	from ..common.rigid_body_payload import RigidBodyLoadAttachmentSet
except ImportError:
	from common.rigid_body_payload import RigidBodyLoadAttachmentSet

from .safe_corridor import CorridorGenerationConfig, InflatedObstaclePrismXZ, InflatedObstacleSphere, SceneObstacle, build_axis_aligned_corridors_from_reference, evaluate_cable_obstacle_collisions, evaluate_obstacle_collisions
from .tension_feasibility import TensionFeasibilityInput, TensionFeasibilityResult, compute_cable_directions_from_uav_positions, solve_tension_feasibility
from .uav_corridor_qp import LocalQPProblem, LocalQPResult, UAVKinematicLimits, solve_local_corridor_qp


def _validate_payload_positions(payload_positions_m: np.ndarray) -> np.ndarray:
	array = np.asarray(payload_positions_m, dtype=float)
	if array.ndim != 2 or array.shape[1] != 3:
		raise ValueError("payload_positions_m must have shape (N, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError("payload_positions_m must be finite")
	return array


def _validate_rotation_series(payload_rotation_matrices: np.ndarray) -> np.ndarray:
	array = np.asarray(payload_rotation_matrices, dtype=float)
	if array.ndim != 3 or array.shape[1:] != (3, 3):
		raise ValueError("payload_rotation_matrices must have shape (N, 3, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError("payload_rotation_matrices must be finite")
	return array


def _validate_reference_positions(reference_uav_positions_m: np.ndarray) -> np.ndarray:
	array = np.asarray(reference_uav_positions_m, dtype=float)
	if array.ndim != 3 or array.shape[2] != 3:
		raise ValueError("reference_uav_positions_m must have shape (N, M, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError("reference_uav_positions_m must be finite")
	return array


@dataclass(frozen=True)
class SequentialPlanningConfig:
	"""Configuration bundle for the first sequential planning scaffold."""

	corridor_config: CorridorGenerationConfig
	kinematic_limits: UAVKinematicLimits
	tension_max_n: np.ndarray
	obstacles: tuple[SceneObstacle, ...] = field(default_factory=tuple)
	max_iterations: int = 1
	length_tolerance_m: float = 5e-3
	residual_tolerance: float = 1e-5
	force_residual_absolute_tolerance_n: float = 0.5
	force_residual_relative_tolerance: float = 0.05
	torque_residual_absolute_tolerance_nm: float = 0.12
	torque_residual_relative_tolerance: float = 2.50
	force_relative_scale_floor_n: float = 1.0
	torque_relative_scale_floor_nm: float = 0.05
	local_reference_weight: float = 1.0
	tension_surrogate_weight: float = 0.25
	tension_linearization_weight: float = 0.50
	local_shape_weight: float = 0.05
	local_smoothness_weight: float = 1.0
	min_separation_m: float = 0.0
	trust_region_half_width_m: float = 0.05
	tension_linearization_delta_m: float = 1e-3
	feedback_force_gain_m_per_n: float = 0.0
	feedback_torque_gain_m_per_nm: float = 0.10
	feedback_max_step_m: float = 0.05
	residual_improvement_tolerance: float = 1e-6
	collision_improvement_tolerance_m: float = 1e-6
	cable_collision_num_samples: int = 9

	def __post_init__(self) -> None:
		tension_max_n = np.asarray(self.tension_max_n, dtype=float).reshape(-1)
		if tension_max_n.ndim != 1 or tension_max_n.size == 0 or np.any(tension_max_n <= 0.0):
			raise ValueError("tension_max_n must be a positive 1D vector")
		if not isinstance(self.corridor_config, CorridorGenerationConfig):
			raise TypeError("corridor_config must be a CorridorGenerationConfig")
		if not isinstance(self.kinematic_limits, UAVKinematicLimits):
			raise TypeError("kinematic_limits must be a UAVKinematicLimits")
		for obstacle in self.obstacles:
			if not isinstance(obstacle, (InflatedObstacleSphere, InflatedObstaclePrismXZ)):
				raise TypeError("obstacles must contain supported obstacle instances")
		if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
			raise ValueError("max_iterations must be a positive integer")
		if not np.isfinite(self.length_tolerance_m) or self.length_tolerance_m < 0.0:
			raise ValueError("length_tolerance_m must be finite and non-negative")
		if not np.isfinite(self.residual_tolerance) or self.residual_tolerance < 0.0:
			raise ValueError("residual_tolerance must be finite and non-negative")
		if not np.isfinite(self.force_residual_absolute_tolerance_n) or self.force_residual_absolute_tolerance_n < 0.0:
			raise ValueError("force_residual_absolute_tolerance_n must be finite and non-negative")
		if not np.isfinite(self.force_residual_relative_tolerance) or self.force_residual_relative_tolerance < 0.0:
			raise ValueError("force_residual_relative_tolerance must be finite and non-negative")
		if not np.isfinite(self.torque_residual_absolute_tolerance_nm) or self.torque_residual_absolute_tolerance_nm < 0.0:
			raise ValueError("torque_residual_absolute_tolerance_nm must be finite and non-negative")
		if not np.isfinite(self.torque_residual_relative_tolerance) or self.torque_residual_relative_tolerance < 0.0:
			raise ValueError("torque_residual_relative_tolerance must be finite and non-negative")
		if not np.isfinite(self.force_relative_scale_floor_n) or self.force_relative_scale_floor_n <= 0.0:
			raise ValueError("force_relative_scale_floor_n must be finite and positive")
		if not np.isfinite(self.torque_relative_scale_floor_nm) or self.torque_relative_scale_floor_nm <= 0.0:
			raise ValueError("torque_relative_scale_floor_nm must be finite and positive")
		if not np.isfinite(self.local_reference_weight) or self.local_reference_weight <= 0.0:
			raise ValueError("local_reference_weight must be finite and positive")
		if not np.isfinite(self.tension_surrogate_weight) or self.tension_surrogate_weight < 0.0:
			raise ValueError("tension_surrogate_weight must be finite and non-negative")
		if not np.isfinite(self.tension_linearization_weight) or self.tension_linearization_weight < 0.0:
			raise ValueError("tension_linearization_weight must be finite and non-negative")
		if not np.isfinite(self.local_shape_weight) or self.local_shape_weight < 0.0:
			raise ValueError("local_shape_weight must be finite and non-negative")
		if not np.isfinite(self.local_smoothness_weight) or self.local_smoothness_weight < 0.0:
			raise ValueError("local_smoothness_weight must be finite and non-negative")
		if not np.isfinite(self.min_separation_m) or self.min_separation_m < 0.0:
			raise ValueError("min_separation_m must be finite and non-negative")
		if not np.isfinite(self.trust_region_half_width_m) or self.trust_region_half_width_m < 0.0:
			raise ValueError("trust_region_half_width_m must be finite and non-negative")
		if not np.isfinite(self.tension_linearization_delta_m) or self.tension_linearization_delta_m <= 0.0:
			raise ValueError("tension_linearization_delta_m must be finite and positive")
		if not np.isfinite(self.feedback_force_gain_m_per_n) or self.feedback_force_gain_m_per_n < 0.0:
			raise ValueError("feedback_force_gain_m_per_n must be finite and non-negative")
		if not np.isfinite(self.feedback_torque_gain_m_per_nm) or self.feedback_torque_gain_m_per_nm < 0.0:
			raise ValueError("feedback_torque_gain_m_per_nm must be finite and non-negative")
		if not np.isfinite(self.feedback_max_step_m) or self.feedback_max_step_m < 0.0:
			raise ValueError("feedback_max_step_m must be finite and non-negative")
		if not np.isfinite(self.residual_improvement_tolerance) or self.residual_improvement_tolerance < 0.0:
			raise ValueError("residual_improvement_tolerance must be finite and non-negative")
		if not np.isfinite(self.collision_improvement_tolerance_m) or self.collision_improvement_tolerance_m < 0.0:
			raise ValueError("collision_improvement_tolerance_m must be finite and non-negative")
		if not isinstance(self.cable_collision_num_samples, int) or self.cable_collision_num_samples < 2:
			raise ValueError("cable_collision_num_samples must be an integer >= 2")


@dataclass(frozen=True)
class SequentialPlanningResult:
	"""First-pass result of the collision-first sequential planner."""

	positions_m: np.ndarray
	qp_result: LocalQPResult
	tension_results: list[TensionFeasibilityResult]
	all_tension_feasible: bool
	all_tension_strictly_feasible: bool
	mean_residual_norm: float
	obstacle_collision_count: int
	min_obstacle_clearance_m: float
	cable_collision_count: int
	min_cable_clearance_m: float
	iterations_run: int
	status: str
	residual_history: np.ndarray
	strict_feasible_history: np.ndarray
	relaxed_feasible_history: np.ndarray
	collision_count_history: np.ndarray
	min_clearance_history_m: np.ndarray
	cable_collision_count_history: np.ndarray
	min_cable_clearance_history_m: np.ndarray


def _compute_anchor_positions_world(
	payload_positions_m: np.ndarray,
	payload_rotation_matrices: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
) -> np.ndarray:
	anchor_positions_m = np.zeros((payload_positions_m.shape[0], attachments.count, 3), dtype=float)
	for sample_index in range(payload_positions_m.shape[0]):
		rotation = np.asarray(payload_rotation_matrices[sample_index], dtype=float)
		anchor_positions_m[sample_index] = payload_positions_m[sample_index][None, :] + (rotation @ attachments.r_i_body_m.T).T
	return anchor_positions_m


def _summarize_tension_results(tension_results: list[TensionFeasibilityResult]) -> tuple[bool, bool, float, int, int, str]:
	all_tension_feasible = bool(all(result.feasible for result in tension_results))
	all_tension_strictly_feasible = bool(all(result.strictly_feasible for result in tension_results))
	mean_residual_norm = float(np.mean([result.residual_norm for result in tension_results]))
	strict_count = int(sum(result.strictly_feasible for result in tension_results))
	relaxed_count = int(sum(result.relaxed_feasible for result in tension_results))
	if all_tension_strictly_feasible:
		status = "reference_strictly_feasible"
	elif all_tension_feasible:
		status = "reference_relaxed_feasible"
	else:
		status = "reference_infeasible"
	return all_tension_feasible, all_tension_strictly_feasible, mean_residual_norm, strict_count, relaxed_count, status


def _build_feedback_reference_update(
	positions_m: np.ndarray,
	payload_positions_m: np.ndarray,
	payload_rotation_matrices: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
	tension_results: list[TensionFeasibilityResult],
	config: SequentialPlanningConfig,
) -> np.ndarray:
	updated_reference = np.asarray(positions_m, dtype=float).copy()
	for sample_index, tension_result in enumerate(tension_results):
		if tension_result.strictly_feasible:
			continue
		rotation = np.asarray(payload_rotation_matrices[sample_index], dtype=float)
		residual_force_world = rotation @ np.asarray(tension_result.residual_wrench_body[:3], dtype=float)
		residual_torque_world = rotation @ np.asarray(tension_result.residual_wrench_body[3:], dtype=float)
		payload_position = np.asarray(payload_positions_m[sample_index], dtype=float)
		anchor_world = payload_position[None, :] + (rotation @ attachments.r_i_body_m.T).T
		for uav_index in range(attachments.count):
			anchor_offset_world = anchor_world[uav_index] - payload_position
			feedback = (
				-config.feedback_force_gain_m_per_n * residual_force_world
				+ config.feedback_torque_gain_m_per_nm * np.cross(residual_torque_world, anchor_offset_world)
			)
			feedback_norm = float(np.linalg.norm(feedback))
			if feedback_norm > config.feedback_max_step_m > 0.0:
				feedback = feedback * (config.feedback_max_step_m / feedback_norm)
			updated_reference[sample_index, uav_index] = updated_reference[sample_index, uav_index] + feedback
	return updated_reference


def _solve_tension_results_for_positions(
	positions_m: np.ndarray,
	time_s: np.ndarray,
	payload_positions_m: np.ndarray,
	payload_rotation_matrices: np.ndarray,
	wrench_body_series: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
	config: SequentialPlanningConfig,
) -> list[TensionFeasibilityResult]:
	tension_results: list[TensionFeasibilityResult] = []
	for sample_index in range(time_s.size):
		tension_results.append(
			solve_tension_feasibility(
				TensionFeasibilityInput(
					payload_position_m=payload_positions_m[sample_index],
					payload_rotation_matrix=payload_rotation_matrices[sample_index],
					uav_positions_m=positions_m[sample_index],
					wrench_body=wrench_body_series[sample_index],
					attachments=attachments,
					tension_max_n=np.asarray(config.tension_max_n, dtype=float),
					length_tolerance_m=config.length_tolerance_m,
					residual_tolerance=config.residual_tolerance,
					force_residual_absolute_tolerance_n=config.force_residual_absolute_tolerance_n,
					force_residual_relative_tolerance=config.force_residual_relative_tolerance,
					torque_residual_absolute_tolerance_nm=config.torque_residual_absolute_tolerance_nm,
					torque_residual_relative_tolerance=config.torque_residual_relative_tolerance,
					force_relative_scale_floor_n=config.force_relative_scale_floor_n,
					torque_relative_scale_floor_nm=config.torque_relative_scale_floor_nm,
				)
			)
		)
	return tension_results


def _build_tension_surrogate_sample_weights(
	tension_results: list[TensionFeasibilityResult],
	base_weight: float,
) -> np.ndarray:
	if base_weight <= 0.0:
		return np.zeros(len(tension_results), dtype=float)
	residuals = np.asarray([result.residual_norm for result in tension_results], dtype=float)
	residual_scale = max(float(np.max(residuals)), 1e-9)
	weights = base_weight * np.clip(residuals / residual_scale, 0.0, 1.0)
	return weights


def _evaluate_fixed_tension_residual_wrench(
	payload_position_m: np.ndarray,
	payload_rotation_matrix: np.ndarray,
	uav_positions_m: np.ndarray,
	wrench_body: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
	tensions_n: np.ndarray,
) -> np.ndarray:
	q_body, _ = compute_cable_directions_from_uav_positions(
		payload_position_m=payload_position_m,
		payload_rotation_matrix=payload_rotation_matrix,
		uav_positions_m=uav_positions_m,
		attachments=attachments,
	)
	reconstructed_wrench_body = attachments.wrench_map_body(q_body) @ np.asarray(tensions_n, dtype=float)
	return np.asarray(wrench_body, dtype=float) - reconstructed_wrench_body


def _build_linearized_tension_terms(
	reference_positions_m: np.ndarray,
	time_s: np.ndarray,
	payload_positions_m: np.ndarray,
	payload_rotation_matrices: np.ndarray,
	wrench_body_series: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
	reference_tension_results: list[TensionFeasibilityResult],
	config: SequentialPlanningConfig,
) -> tuple[np.ndarray, np.ndarray]:
	n_sample, n_uav, _ = reference_positions_m.shape
	residuals = np.zeros((n_sample, 6), dtype=float)
	jacobians = np.zeros((n_sample, 6, n_uav, 3), dtype=float)
	finite_difference_step = float(config.tension_linearization_delta_m)
	for sample_index in range(time_s.size):
		reference_positions_sample = np.asarray(reference_positions_m[sample_index], dtype=float)
		tensions_sample = np.asarray(reference_tension_results[sample_index].tensions_n, dtype=float)
		residual_sample = _evaluate_fixed_tension_residual_wrench(
			payload_position_m=payload_positions_m[sample_index],
			payload_rotation_matrix=payload_rotation_matrices[sample_index],
			uav_positions_m=reference_positions_sample,
			wrench_body=wrench_body_series[sample_index],
			attachments=attachments,
			tensions_n=tensions_sample,
		)
		residuals[sample_index] = residual_sample
		for uav_index in range(n_uav):
			for axis_index in range(3):
				perturbed_positions = reference_positions_sample.copy()
				perturbed_positions[uav_index, axis_index] += finite_difference_step
				perturbed_residual = _evaluate_fixed_tension_residual_wrench(
					payload_position_m=payload_positions_m[sample_index],
					payload_rotation_matrix=payload_rotation_matrices[sample_index],
					uav_positions_m=perturbed_positions,
					wrench_body=wrench_body_series[sample_index],
					attachments=attachments,
					tensions_n=tensions_sample,
				)
				jacobians[sample_index, :, uav_index, axis_index] = (perturbed_residual - residual_sample) / finite_difference_step
	return residuals, jacobians


def plan_collision_aware_uav_positions(
	time_s: np.ndarray,
	payload_positions_m: np.ndarray,
	payload_rotation_matrices: np.ndarray,
	reference_uav_positions_m: np.ndarray,
	wrench_body_series: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
	config: SequentialPlanningConfig,
) -> SequentialPlanningResult:
	"""Run the first scaffold of the collision-first UAV pose planner."""

	time_s = np.asarray(time_s, dtype=float).reshape(-1)
	payload_positions_m = _validate_payload_positions(payload_positions_m)
	payload_rotation_matrices = _validate_rotation_series(payload_rotation_matrices)
	reference_uav_positions_m = _validate_reference_positions(reference_uav_positions_m)
	wrench_body_series = np.asarray(wrench_body_series, dtype=float)
	if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
		raise ValueError("time_s must be finite and strictly increasing")
	if payload_positions_m.shape[0] != time_s.size:
		raise ValueError("payload_positions_m must align with time_s")
	if payload_rotation_matrices.shape[0] != time_s.size:
		raise ValueError("payload_rotation_matrices must align with time_s")
	if reference_uav_positions_m.shape[0] != time_s.size or reference_uav_positions_m.shape[1] != attachments.count:
		raise ValueError("reference_uav_positions_m must align with time_s and attachments")
	if wrench_body_series.shape != (time_s.size, 6) or not np.all(np.isfinite(wrench_body_series)):
		raise ValueError("wrench_body_series must have shape (N, 6)")
	if np.asarray(config.tension_max_n, dtype=float).shape != (attachments.count,):
		raise ValueError("config.tension_max_n must align with attachments")
	current_reference_positions_m = np.asarray(reference_uav_positions_m, dtype=float).copy()
	best_result: SequentialPlanningResult | None = None
	best_score: tuple[int, int, float, float, int, int, float] | None = None
	residual_history: list[float] = []
	strict_history: list[int] = []
	relaxed_history: list[int] = []
	collision_count_history: list[int] = []
	min_clearance_history_m: list[float] = []
	cable_collision_count_history: list[int] = []
	min_cable_clearance_history_m: list[float] = []
	anchor_positions_m = _compute_anchor_positions_world(
		payload_positions_m=payload_positions_m,
		payload_rotation_matrices=payload_rotation_matrices,
		attachments=attachments,
	)

	for iteration_index in range(config.max_iterations):
		tension_surrogate_positions_m = None
		tension_surrogate_sample_weights = None
		tension_linearized_residuals = None
		tension_linearized_jacobians = None
		reference_tension_results: list[TensionFeasibilityResult] | None = None
		if config.tension_surrogate_weight > 0.0 or config.tension_linearization_weight > 0.0:
			reference_tension_results = _solve_tension_results_for_positions(
				positions_m=current_reference_positions_m,
				time_s=time_s,
				payload_positions_m=payload_positions_m,
				payload_rotation_matrices=payload_rotation_matrices,
				wrench_body_series=wrench_body_series,
				attachments=attachments,
				config=config,
			)
		if config.tension_linearization_weight > 0.0 and reference_tension_results is not None:
			tension_linearized_residuals, tension_linearized_jacobians = _build_linearized_tension_terms(
				reference_positions_m=current_reference_positions_m,
				time_s=time_s,
				payload_positions_m=payload_positions_m,
				payload_rotation_matrices=payload_rotation_matrices,
				wrench_body_series=wrench_body_series,
				attachments=attachments,
				reference_tension_results=reference_tension_results,
				config=config,
			)
		if config.tension_surrogate_weight > 0.0 and reference_tension_results is not None:
			tension_surrogate_positions_m = _build_feedback_reference_update(
				positions_m=current_reference_positions_m,
				payload_positions_m=payload_positions_m,
				payload_rotation_matrices=payload_rotation_matrices,
				attachments=attachments,
				tension_results=reference_tension_results,
				config=config,
			)
			tension_surrogate_sample_weights = _build_tension_surrogate_sample_weights(
				tension_results=reference_tension_results,
				base_weight=1.0,
			)
		corridors = build_axis_aligned_corridors_from_reference(
			reference_positions_m=current_reference_positions_m,
			config=config.corridor_config,
			obstacles=list(config.obstacles),
		)
		qp_result = solve_local_corridor_qp(
			LocalQPProblem(
				time_s=time_s,
				reference_positions_m=current_reference_positions_m,
				corridors=corridors,
				kinematic_limits=config.kinematic_limits,
				reference_weight=config.local_reference_weight,
				tension_surrogate_positions_m=tension_surrogate_positions_m,
				tension_surrogate_sample_weights=tension_surrogate_sample_weights,
				tension_surrogate_weight=config.tension_surrogate_weight,
				tension_linearized_residuals=tension_linearized_residuals,
				tension_linearized_jacobians=tension_linearized_jacobians,
				tension_linearization_weight=config.tension_linearization_weight,
				shape_weight=config.local_shape_weight,
				smoothness_weight=config.local_smoothness_weight,
				min_separation_m=config.min_separation_m,
				trust_region_half_width_m=config.trust_region_half_width_m,
			)
		)

		tension_results = _solve_tension_results_for_positions(
			positions_m=qp_result.positions_m,
			time_s=time_s,
			payload_positions_m=payload_positions_m,
			payload_rotation_matrices=payload_rotation_matrices,
			wrench_body_series=wrench_body_series,
			attachments=attachments,
			config=config,
		)

		all_tension_feasible, all_tension_strictly_feasible, mean_residual_norm, strict_count, relaxed_count, status = _summarize_tension_results(tension_results)
		obstacle_collision_count = 0
		min_obstacle_clearance_m = float("nan")
		cable_collision_count = 0
		min_cable_clearance_m = float("nan")
		if config.obstacles:
			obstacle_collision_count, min_obstacle_clearance_m = evaluate_obstacle_collisions(qp_result.positions_m, config.obstacles)
			cable_collision_count, min_cable_clearance_m = evaluate_cable_obstacle_collisions(
				anchor_positions_m=anchor_positions_m,
				uav_positions_m=qp_result.positions_m,
				obstacles=config.obstacles,
				n_samples_per_cable=config.cable_collision_num_samples,
			)
		residual_history.append(mean_residual_norm)
		strict_history.append(strict_count)
		relaxed_history.append(relaxed_count)
		collision_count_history.append(obstacle_collision_count)
		min_clearance_history_m.append(min_obstacle_clearance_m)
		cable_collision_count_history.append(cable_collision_count)
		min_cable_clearance_history_m.append(min_cable_clearance_m)
		candidate_result = SequentialPlanningResult(
			positions_m=qp_result.positions_m,
			qp_result=qp_result,
			tension_results=tension_results,
			all_tension_feasible=all_tension_feasible,
			all_tension_strictly_feasible=all_tension_strictly_feasible,
			mean_residual_norm=mean_residual_norm,
			obstacle_collision_count=obstacle_collision_count,
			min_obstacle_clearance_m=min_obstacle_clearance_m,
			cable_collision_count=cable_collision_count,
			min_cable_clearance_m=min_cable_clearance_m,
			iterations_run=iteration_index + 1,
			status=status,
			residual_history=np.asarray(residual_history, dtype=float),
			strict_feasible_history=np.asarray(strict_history, dtype=int),
			relaxed_feasible_history=np.asarray(relaxed_history, dtype=int),
			collision_count_history=np.asarray(collision_count_history, dtype=int),
			min_clearance_history_m=np.asarray(min_clearance_history_m, dtype=float),
			cable_collision_count_history=np.asarray(cable_collision_count_history, dtype=int),
			min_cable_clearance_history_m=np.asarray(min_cable_clearance_history_m, dtype=float),
		)

		candidate_score = (-cable_collision_count, -obstacle_collision_count, min_cable_clearance_m, min_obstacle_clearance_m, strict_count, relaxed_count, -mean_residual_norm)
		if best_result is None or best_score is None or candidate_score > best_score:
			best_result = candidate_result
			best_score = candidate_score

		if all_tension_strictly_feasible and obstacle_collision_count == 0 and cable_collision_count == 0:
			break
		if iteration_index == config.max_iterations - 1:
			break

		current_reference_positions_m = _build_feedback_reference_update(
			positions_m=qp_result.positions_m,
			payload_positions_m=payload_positions_m,
			payload_rotation_matrices=payload_rotation_matrices,
			attachments=attachments,
			tension_results=tension_results,
			config=config,
		)

		if best_result is not None and len(residual_history) >= 2:
			collision_count_improved = collision_count_history[-1] < collision_count_history[-2]
			clearance_improved = min_clearance_history_m[-1] > min_clearance_history_m[-2] + config.collision_improvement_tolerance_m
			cable_collision_improved = cable_collision_count_history[-1] < cable_collision_count_history[-2]
			cable_clearance_improved = min_cable_clearance_history_m[-1] > min_cable_clearance_history_m[-2] + config.collision_improvement_tolerance_m
			tension_improved = residual_history[-2] - residual_history[-1] >= config.residual_improvement_tolerance
			feasibility_improved = strict_history[-1] > strict_history[-2] or relaxed_history[-1] > relaxed_history[-2]
			if not (collision_count_improved or clearance_improved or cable_collision_improved or cable_clearance_improved or tension_improved or feasibility_improved):
				break

	if best_result is None:
		raise RuntimeError("sequential planner did not produce any candidate result")
	return SequentialPlanningResult(
		positions_m=best_result.positions_m,
		qp_result=best_result.qp_result,
		tension_results=best_result.tension_results,
		all_tension_feasible=best_result.all_tension_feasible,
		all_tension_strictly_feasible=best_result.all_tension_strictly_feasible,
		mean_residual_norm=best_result.mean_residual_norm,
		obstacle_collision_count=best_result.obstacle_collision_count,
		min_obstacle_clearance_m=best_result.min_obstacle_clearance_m,
		cable_collision_count=best_result.cable_collision_count,
		min_cable_clearance_m=best_result.min_cable_clearance_m,
		iterations_run=len(residual_history),
		status=best_result.status,
		residual_history=np.asarray(residual_history, dtype=float),
		strict_feasible_history=np.asarray(strict_history, dtype=int),
		relaxed_feasible_history=np.asarray(relaxed_history, dtype=int),
		collision_count_history=np.asarray(collision_count_history, dtype=int),
		min_clearance_history_m=np.asarray(min_clearance_history_m, dtype=float),
		cable_collision_count_history=np.asarray(cable_collision_count_history, dtype=int),
		min_cable_clearance_history_m=np.asarray(min_cable_clearance_history_m, dtype=float),
	)