# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-10
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-10
# 目的 / Purpose:
#   定义局部安全走廊 QP 的输入输出接口，并提供第一版“参考轨迹透传 + 诊断”
#   的最小实现，供顺序规划主循环先跑通。
#   Define the local safe-corridor QP interfaces and provide a first-pass
#   "reference pass-through + diagnostics" implementation so the sequential
#   planning loop can already run end-to-end.
# 主要输入 / Main Inputs:
#   参考 UAV 轨迹、走廊约束、离散速度/加速度边界。
#   Reference UAV trajectories, corridor constraints, and discrete kinematic limits.
# 主要输出 / Main Outputs:
#   更新后的 UAV 轨迹、走廊违反统计、运动学诊断。
#   Updated UAV trajectories, corridor violation summaries, and kinematic diagnostics.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import osqp

from .safe_corridor import HalfspaceCorridor3D


def _validate_reference_positions(reference_positions_m: np.ndarray) -> np.ndarray:
	array = np.asarray(reference_positions_m, dtype=float)
	if array.ndim != 3 or array.shape[2] != 3:
		raise ValueError("reference_positions_m must have shape (N, M, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError("reference_positions_m must be finite")
	return array


@dataclass(frozen=True)
class UAVKinematicLimits:
	"""Simple discrete kinematic bounds for the UAV position trajectory."""

	max_speed_mps: float
	max_acceleration_mps2: float

	def __post_init__(self) -> None:
		if not np.isfinite(self.max_speed_mps) or self.max_speed_mps <= 0.0:
			raise ValueError("max_speed_mps must be finite and positive")
		if not np.isfinite(self.max_acceleration_mps2) or self.max_acceleration_mps2 <= 0.0:
			raise ValueError("max_acceleration_mps2 must be finite and positive")


@dataclass(frozen=True)
class LocalQPProblem:
	"""Problem bundle for one local safe-corridor trajectory update."""

	time_s: np.ndarray
	reference_positions_m: np.ndarray
	corridors: list[list[HalfspaceCorridor3D]]
	kinematic_limits: UAVKinematicLimits
	reference_weight: float = 1.0
	tension_surrogate_positions_m: np.ndarray | None = None
	tension_surrogate_sample_weights: np.ndarray | None = None
	tension_surrogate_weight: float = 0.0
	tension_linearized_residuals: np.ndarray | None = None
	tension_linearized_jacobians: np.ndarray | None = None
	tension_linearization_weight: float = 0.0
	shape_weight: float = 0.0
	smoothness_weight: float = 1.0
	min_separation_m: float = 0.0
	trust_region_half_width_m: float | None = None
	constraint_tolerance_m: float = 1e-6

	def __post_init__(self) -> None:
		time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
		reference_positions_m = _validate_reference_positions(self.reference_positions_m)
		if time_s.ndim != 1 or time_s.size != reference_positions_m.shape[0]:
			raise ValueError("time_s must align with reference_positions_m")
		if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
			raise ValueError("time_s must be finite and strictly increasing")
		if len(self.corridors) != reference_positions_m.shape[0]:
			raise ValueError("corridors must align with time_s")
		for sample_index, sample_corridors in enumerate(self.corridors):
			if len(sample_corridors) != reference_positions_m.shape[1]:
				raise ValueError(f"corridors[{sample_index}] must align with UAV count")
		if not np.isfinite(self.reference_weight) or self.reference_weight <= 0.0:
			raise ValueError("reference_weight must be finite and positive")
		if self.tension_surrogate_positions_m is not None:
			tension_surrogate_positions_m = _validate_reference_positions(self.tension_surrogate_positions_m)
			if tension_surrogate_positions_m.shape != reference_positions_m.shape:
				raise ValueError("tension_surrogate_positions_m must align with reference_positions_m")
		if self.tension_surrogate_sample_weights is not None:
			tension_surrogate_sample_weights = np.asarray(self.tension_surrogate_sample_weights, dtype=float).reshape(-1)
			if tension_surrogate_sample_weights.shape != (reference_positions_m.shape[0],):
				raise ValueError("tension_surrogate_sample_weights must have shape (N,)")
			if not np.all(np.isfinite(tension_surrogate_sample_weights)) or np.any(tension_surrogate_sample_weights < 0.0):
				raise ValueError("tension_surrogate_sample_weights must be finite and non-negative")
		if not np.isfinite(self.tension_surrogate_weight) or self.tension_surrogate_weight < 0.0:
			raise ValueError("tension_surrogate_weight must be finite and non-negative")
		if self.tension_linearized_residuals is not None:
			tension_linearized_residuals = np.asarray(self.tension_linearized_residuals, dtype=float)
			if tension_linearized_residuals.shape != (reference_positions_m.shape[0], 6) or not np.all(np.isfinite(tension_linearized_residuals)):
				raise ValueError("tension_linearized_residuals must have shape (N, 6) and be finite")
		if self.tension_linearized_jacobians is not None:
			tension_linearized_jacobians = np.asarray(self.tension_linearized_jacobians, dtype=float)
			expected_shape = (reference_positions_m.shape[0], 6, reference_positions_m.shape[1], 3)
			if tension_linearized_jacobians.shape != expected_shape or not np.all(np.isfinite(tension_linearized_jacobians)):
				raise ValueError("tension_linearized_jacobians must have shape (N, 6, M, 3) and be finite")
		if (self.tension_linearized_residuals is None) != (self.tension_linearized_jacobians is None):
			raise ValueError("tension_linearized_residuals and tension_linearized_jacobians must be provided together")
		if not np.isfinite(self.tension_linearization_weight) or self.tension_linearization_weight < 0.0:
			raise ValueError("tension_linearization_weight must be finite and non-negative")
		if not np.isfinite(self.shape_weight) or self.shape_weight < 0.0:
			raise ValueError("shape_weight must be finite and non-negative")
		if not np.isfinite(self.smoothness_weight) or self.smoothness_weight < 0.0:
			raise ValueError("smoothness_weight must be finite and non-negative")
		if not np.isfinite(self.min_separation_m) or self.min_separation_m < 0.0:
			raise ValueError("min_separation_m must be finite and non-negative")
		if self.trust_region_half_width_m is not None:
			if not np.isfinite(self.trust_region_half_width_m) or self.trust_region_half_width_m < 0.0:
				raise ValueError("trust_region_half_width_m must be finite and non-negative when provided")
		if not np.isfinite(self.constraint_tolerance_m) or self.constraint_tolerance_m < 0.0:
			raise ValueError("constraint_tolerance_m must be finite and non-negative")


@dataclass(frozen=True)
class LocalQPResult:
	"""Result of a local safe-corridor trajectory update."""

	positions_m: np.ndarray
	corridor_max_violation_m: float
	separation_min_margin_m: float
	max_speed_mps: float
	max_acceleration_mps2: float
	status: str


def evaluate_discrete_kinematics(time_s: np.ndarray, positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Compute discrete velocity and acceleration for a position trajectory."""

	time_s = np.asarray(time_s, dtype=float).reshape(-1)
	positions_m = _validate_reference_positions(positions_m)
	if time_s.size < 2:
		zero = np.zeros_like(positions_m, dtype=float)
		return zero, zero
	velocity_mps = np.gradient(positions_m, time_s, axis=0, edge_order=1)
	acceleration_mps2 = np.gradient(velocity_mps, time_s, axis=0, edge_order=1)
	return velocity_mps, acceleration_mps2


def _flat_index(sample_index: int, uav_index: int, axis_index: int, n_uav: int) -> int:
	return (sample_index * n_uav + uav_index) * 3 + axis_index


def _build_second_difference_matrix(n_sample: int) -> sp.csc_matrix | None:
	if n_sample < 3:
		return None
	rows = []
	cols = []
	data = []
	for row in range(n_sample - 2):
		rows.extend([row, row, row])
		cols.extend([row, row + 1, row + 2])
		data.extend([1.0, -2.0, 1.0])
	return sp.coo_matrix((data, (rows, cols)), shape=(n_sample - 2, n_sample), dtype=float).tocsc()


def _build_sample_diagonal(weights: np.ndarray, n_uav: int) -> sp.csc_matrix:
	entries = np.repeat(np.asarray(weights, dtype=float), n_uav * 3)
	return sp.diags(entries, format="csc")


def _build_shape_difference_matrix(n_sample: int, n_uav: int) -> sp.csc_matrix | None:
	if n_uav < 2:
		return None
	rows = []
	cols = []
	data = []
	row_index = 0
	for sample_index in range(n_sample):
		for i in range(n_uav):
			for j in range(i + 1, n_uav):
				for axis_index in range(3):
					rows.extend([row_index, row_index])
					cols.extend([
						_flat_index(sample_index, i, axis_index, n_uav),
						_flat_index(sample_index, j, axis_index, n_uav),
					])
					data.extend([1.0, -1.0])
					row_index += 1
	if row_index == 0:
		return None
	return sp.coo_matrix((data, (rows, cols)), shape=(row_index, n_sample * n_uav * 3), dtype=float).tocsc()


def _evaluate_min_separation_margin(positions_m: np.ndarray, min_separation_m: float) -> float:
	if min_separation_m <= 0.0:
		return float("inf")
	positions_m = _validate_reference_positions(positions_m)
	margin = float("inf")
	for sample_index in range(positions_m.shape[0]):
		for i in range(positions_m.shape[1]):
			for j in range(i + 1, positions_m.shape[1]):
				distance = float(np.linalg.norm(positions_m[sample_index, i] - positions_m[sample_index, j]))
				margin = min(margin, distance - min_separation_m)
	return margin


def solve_local_corridor_qp(problem: LocalQPProblem) -> LocalQPResult:
	"""Solve a discrete local corridor QP around the current reference trajectory."""

	time_s = np.asarray(problem.time_s, dtype=float).reshape(-1)
	reference_positions_m = np.asarray(problem.reference_positions_m, dtype=float)
	n_sample, n_uav, _ = reference_positions_m.shape
	n_var = n_sample * n_uav * 3
	x_ref = reference_positions_m.reshape(-1)

	P = sp.eye(n_var, format="csc") * float(problem.reference_weight)
	q = -float(problem.reference_weight) * x_ref.copy()
	if problem.tension_surrogate_positions_m is not None and problem.tension_surrogate_weight > 0.0:
		x_tension = np.asarray(problem.tension_surrogate_positions_m, dtype=float).reshape(-1)
		if problem.tension_surrogate_sample_weights is None:
			W_tension = sp.eye(n_var, format="csc") * float(problem.tension_surrogate_weight)
		else:
			W_tension = _build_sample_diagonal(problem.tension_surrogate_sample_weights, n_uav) * float(problem.tension_surrogate_weight)
		P = P + W_tension
		q = q - W_tension @ x_tension

	if problem.tension_linearized_residuals is not None and problem.tension_linearized_jacobians is not None and problem.tension_linearization_weight > 0.0:
		P_tension_linearized = sp.lil_matrix((n_var, n_var), dtype=float)
		q_tension_linearized = np.zeros(n_var, dtype=float)
		sample_weights = np.ones(n_sample, dtype=float)
		if problem.tension_surrogate_sample_weights is not None:
			sample_weights = np.asarray(problem.tension_surrogate_sample_weights, dtype=float)
		for sample_index in range(n_sample):
			sample_weight = float(problem.tension_linearization_weight * sample_weights[sample_index])
			if sample_weight <= 0.0:
				continue
			jacobian_sample = np.asarray(problem.tension_linearized_jacobians[sample_index], dtype=float).reshape(6, n_uav * 3)
			residual_sample = np.asarray(problem.tension_linearized_residuals[sample_index], dtype=float).reshape(6)
			x_ref_sample = reference_positions_m[sample_index].reshape(-1)
			constant_term = residual_sample - jacobian_sample @ x_ref_sample
			hessian_sample = sample_weight * (jacobian_sample.T @ jacobian_sample)
			gradient_sample = sample_weight * (jacobian_sample.T @ constant_term)
			flat_start = sample_index * n_uav * 3
			flat_stop = flat_start + n_uav * 3
			P_tension_linearized[flat_start:flat_stop, flat_start:flat_stop] += hessian_sample
			q_tension_linearized[flat_start:flat_stop] += gradient_sample
		P = P + P_tension_linearized.tocsc()
		q = q + q_tension_linearized

	shape_diff = _build_shape_difference_matrix(n_sample, n_uav)
	if shape_diff is not None and problem.shape_weight > 0.0:
		shape_ref = shape_diff @ x_ref
		P = P + float(problem.shape_weight) * (shape_diff.T @ shape_diff)
		q = q - float(problem.shape_weight) * (shape_diff.T @ shape_ref)

	second_diff = _build_second_difference_matrix(n_sample)
	if second_diff is not None and problem.smoothness_weight > 0.0:
		axis_penalties = []
		for uav_index in range(n_uav):
			for axis_index in range(3):
				block = sp.lil_matrix((n_sample, n_var), dtype=float)
				for sample_index in range(n_sample):
					block[sample_index, _flat_index(sample_index, uav_index, axis_index, n_uav)] = 1.0
				axis_penalties.append(second_diff @ block.tocsc())
		if axis_penalties:
			D = sp.vstack(axis_penalties, format="csc")
			P = P + float(problem.smoothness_weight) * (D.T @ D)

	A_rows = []
	l_rows = []
	u_rows = []
	tol = float(problem.constraint_tolerance_m)

	def add_leq(row_dict: dict[int, float], ub: float) -> None:
		A_rows.append(row_dict)
		l_rows.append(-np.inf)
		u_rows.append(float(ub))

	for sample_index, sample_corridors in enumerate(problem.corridors):
		for uav_index, corridor in enumerate(sample_corridors):
			for row_index in range(corridor.A.shape[0]):
				row = {}
				for axis_index in range(3):
					value = float(corridor.A[row_index, axis_index])
					if value != 0.0:
						row[_flat_index(sample_index, uav_index, axis_index, n_uav)] = value
				add_leq(row, float(corridor.b[row_index] + tol))

	if problem.trust_region_half_width_m is not None and problem.trust_region_half_width_m > 0.0:
		trust_region_half_width_m = float(problem.trust_region_half_width_m)
		for sample_index in range(n_sample):
			for uav_index in range(n_uav):
				for axis_index in range(3):
					flat_index = _flat_index(sample_index, uav_index, axis_index, n_uav)
					reference_value = float(reference_positions_m[sample_index, uav_index, axis_index])
					add_leq({flat_index: 1.0}, reference_value + trust_region_half_width_m + tol)
					add_leq({flat_index: -1.0}, -(reference_value - trust_region_half_width_m) + tol)

	for sample_index in range(n_sample - 1):
		dt = float(time_s[sample_index + 1] - time_s[sample_index])
		bound = float(problem.kinematic_limits.max_speed_mps * dt + tol)
		for uav_index in range(n_uav):
			for axis_index in range(3):
				idx_now = _flat_index(sample_index, uav_index, axis_index, n_uav)
				idx_next = _flat_index(sample_index + 1, uav_index, axis_index, n_uav)
				add_leq({idx_next: 1.0, idx_now: -1.0}, bound)
				add_leq({idx_next: -1.0, idx_now: 1.0}, bound)

	for sample_index in range(1, n_sample - 1):
		dt_prev = float(time_s[sample_index] - time_s[sample_index - 1])
		dt_next = float(time_s[sample_index + 1] - time_s[sample_index])
		dt_nominal = max(0.5 * (dt_prev + dt_next), 1e-9)
		bound = float(problem.kinematic_limits.max_acceleration_mps2 * (dt_nominal ** 2) + tol)
		for uav_index in range(n_uav):
			for axis_index in range(3):
				idx_prev = _flat_index(sample_index - 1, uav_index, axis_index, n_uav)
				idx_now = _flat_index(sample_index, uav_index, axis_index, n_uav)
				idx_next = _flat_index(sample_index + 1, uav_index, axis_index, n_uav)
				add_leq({idx_prev: 1.0, idx_now: -2.0, idx_next: 1.0}, bound)
				add_leq({idx_prev: -1.0, idx_now: 2.0, idx_next: -1.0}, bound)

	if problem.min_separation_m > 0.0:
		for sample_index in range(n_sample):
			for i in range(n_uav):
				for j in range(i + 1, n_uav):
					delta_ref = reference_positions_m[sample_index, i] - reference_positions_m[sample_index, j]
					norm_ref = float(np.linalg.norm(delta_ref))
					if norm_ref <= 1e-9:
						continue
					normal = delta_ref / norm_ref
					rhs = float(problem.min_separation_m - norm_ref + normal @ delta_ref)
					row = {}
					for axis_index in range(3):
						value = float(normal[axis_index])
						if value != 0.0:
							row[_flat_index(sample_index, i, axis_index, n_uav)] = value
							row[_flat_index(sample_index, j, axis_index, n_uav)] = -value
					add_leq({key: -value for key, value in row.items()}, -rhs + tol)

	A = sp.lil_matrix((len(A_rows), n_var), dtype=float)
	for row_index, row in enumerate(A_rows):
		for col_index, value in row.items():
			A[row_index, col_index] = value
	A = A.tocsc()
	l = np.asarray(l_rows, dtype=float)
	u = np.asarray(u_rows, dtype=float)

	prob = osqp.OSQP()
	prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, max_iter=20000, polishing=True, warm_starting=True)
	res = prob.solve()
	if res.info.status not in ("solved", "solved inaccurate"):
		raise RuntimeError(f"OSQP failed: {res.info.status} (val={res.info.status_val})")

	positions_m = np.asarray(res.x, dtype=float).reshape(n_sample, n_uav, 3)
	corridor_max_violation_m = 0.0
	for sample_index, sample_corridors in enumerate(problem.corridors):
		for uav_index, corridor in enumerate(sample_corridors):
			corridor_max_violation_m = max(
				corridor_max_violation_m,
				corridor.max_violation(positions_m[sample_index, uav_index]),
			)

	velocity_mps, acceleration_mps2 = evaluate_discrete_kinematics(problem.time_s, positions_m)
	max_speed_mps = float(np.max(np.linalg.norm(velocity_mps, axis=2)))
	max_acceleration_mps2 = float(np.max(np.linalg.norm(acceleration_mps2, axis=2)))
	separation_min_margin_m = _evaluate_min_separation_margin(positions_m, problem.min_separation_m)
	return LocalQPResult(
		positions_m=positions_m,
		corridor_max_violation_m=corridor_max_violation_m,
		separation_min_margin_m=separation_min_margin_m,
		max_speed_mps=max_speed_mps,
		max_acceleration_mps2=max_acceleration_mps2,
		status=res.info.status,
	)