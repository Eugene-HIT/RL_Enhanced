# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-10
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-10
# 目的 / Purpose:
#   对给定 UAV 位置、载荷位姿与目标 wrench 执行张力可行性检查，作为碰撞优先
#   外层轨迹优化的内层可行性层。
#   Perform tension-feasibility checks for given UAV positions, payload pose,
#   and target wrench as the inner feasibility layer of the collision-first
#   outer trajectory optimizer.
# 主要输入 / Main Inputs:
#   载荷位姿、UAV 位置、挂点几何、目标 wrench 与张力上界。
#   Payload pose, UAV positions, attachment geometry, target wrench, and
#   tension upper bounds.
# 主要输出 / Main Outputs:
#   绳索方向、张力分配、残差、可行性标记与诊断文本。
#   Cable directions, tension allocation, residual, feasibility flag, and a
#   diagnostic message.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass

import itertools
import numpy as np

try:
	from ..common.rigid_body_payload import RigidBodyLoadAttachmentSet
except ImportError:
	from common.rigid_body_payload import RigidBodyLoadAttachmentSet


def _validate_uav_positions(uav_positions_m: np.ndarray) -> np.ndarray:
	array = np.asarray(uav_positions_m, dtype=float)
	if array.ndim != 2 or array.shape[1] != 3:
		raise ValueError("uav_positions_m must have shape (N, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError("uav_positions_m must be finite")
	return array


def _validate_pose_vector(name: str, value: np.ndarray) -> np.ndarray:
	array = np.asarray(value, dtype=float).reshape(-1)
	if array.shape != (3,) or not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be a finite 3-vector")
	return array


@dataclass(frozen=True)
class TensionFeasibilityInput:
	"""All data required to test one sample of UAV-position feasibility."""

	payload_position_m: np.ndarray
	payload_rotation_matrix: np.ndarray
	uav_positions_m: np.ndarray
	wrench_body: np.ndarray
	attachments: RigidBodyLoadAttachmentSet
	tension_max_n: np.ndarray
	length_tolerance_m: float = 5e-3
	residual_tolerance: float = 1e-5
	force_residual_absolute_tolerance_n: float = 0.5
	force_residual_relative_tolerance: float = 0.05
	torque_residual_absolute_tolerance_nm: float = 0.12
	torque_residual_relative_tolerance: float = 2.50
	force_relative_scale_floor_n: float = 1.0
	torque_relative_scale_floor_nm: float = 0.05

	def __post_init__(self) -> None:
		payload_position_m = _validate_pose_vector("payload_position_m", self.payload_position_m)
		payload_rotation_matrix = np.asarray(self.payload_rotation_matrix, dtype=float)
		uav_positions_m = _validate_uav_positions(self.uav_positions_m)
		wrench_body = np.asarray(self.wrench_body, dtype=float).reshape(-1)
		tension_max_n = np.asarray(self.tension_max_n, dtype=float).reshape(-1)
		if payload_rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(payload_rotation_matrix)):
			raise ValueError("payload_rotation_matrix must have shape (3, 3)")
		if wrench_body.shape != (6,) or not np.all(np.isfinite(wrench_body)):
			raise ValueError("wrench_body must be a finite 6-vector")
		if uav_positions_m.shape[0] != self.attachments.count:
			raise ValueError("uav_positions_m count must match attachments")
		if tension_max_n.shape != (self.attachments.count,) or np.any(tension_max_n <= 0.0):
			raise ValueError("tension_max_n must be positive and aligned with attachments")
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


@dataclass(frozen=True)
class TensionFeasibilityResult:
	"""Result of one tension-feasibility solve or check."""

	feasible: bool
	strictly_feasible: bool
	relaxed_feasible: bool
	message: str
	q_body: np.ndarray
	tensions_n: np.ndarray
	reconstructed_wrench_body: np.ndarray
	residual_wrench_body: np.ndarray
	residual_norm: float
	force_residual_norm: float
	torque_residual_norm: float
	force_relative_residual: float
	torque_relative_residual: float
	cable_length_error_m: np.ndarray
	max_tension_ratio: float


def compute_cable_directions_from_uav_positions(
	payload_position_m: np.ndarray,
	payload_rotation_matrix: np.ndarray,
	uav_positions_m: np.ndarray,
	attachments: RigidBodyLoadAttachmentSet,
) -> tuple[np.ndarray, np.ndarray]:
	"""Recover body-frame cable directions and cable-length mismatch from UAV positions."""

	payload_position_m = _validate_pose_vector("payload_position_m", payload_position_m)
	payload_rotation_matrix = np.asarray(payload_rotation_matrix, dtype=float)
	uav_positions_m = _validate_uav_positions(uav_positions_m)
	if payload_rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(payload_rotation_matrix)):
		raise ValueError("payload_rotation_matrix must have shape (3, 3)")
	if uav_positions_m.shape[0] != attachments.count:
		raise ValueError("uav_positions_m count must match attachments")

	anchor_world_m = payload_position_m[None, :] + (payload_rotation_matrix @ attachments.r_i_body_m.T).T
	cable_vectors_world_m = anchor_world_m - uav_positions_m
	cable_lengths_now = np.linalg.norm(cable_vectors_world_m, axis=1)
	if np.any(cable_lengths_now <= 1e-12):
		raise ValueError("uav_positions_m must not coincide with payload attachment points")

	q_world = cable_vectors_world_m / cable_lengths_now[:, None]
	q_body = (payload_rotation_matrix.T @ q_world.T).T
	length_error_m = cable_lengths_now - np.asarray(attachments.cable_lengths_m, dtype=float)
	return q_body, length_error_m


def _solve_bounded_tension_least_squares(
	phi: np.ndarray,
	wrench_body: np.ndarray,
	tension_max_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	"""Enumerate small active sets to solve bounded least squares without extra dependencies."""

	n_cable = int(tension_max_n.size)
	best_tension = np.zeros(n_cable, dtype=float)
	best_reconstructed = np.zeros(6, dtype=float)
	best_residual = np.asarray(wrench_body, dtype=float).reshape(6)
	best_norm = float(np.linalg.norm(best_residual))
	status_values = ("free", "lower", "upper")

	for statuses in itertools.product(status_values, repeat=n_cable):
		tension = np.zeros(n_cable, dtype=float)
		free_idx = [idx for idx, status in enumerate(statuses) if status == "free"]
		fixed_idx = [idx for idx, status in enumerate(statuses) if status != "free"]

		if fixed_idx:
			for idx in fixed_idx:
				tension[idx] = 0.0 if statuses[idx] == "lower" else tension_max_n[idx]
			reduced_rhs = wrench_body - phi[:, fixed_idx] @ tension[fixed_idx]
		else:
			reduced_rhs = wrench_body

		if free_idx:
			phi_free = phi[:, free_idx]
			free_solution, *_ = np.linalg.lstsq(phi_free, reduced_rhs, rcond=None)
			if np.any(free_solution < -1e-10) or np.any(free_solution > tension_max_n[free_idx] + 1e-10):
				continue
			tension[free_idx] = np.clip(free_solution, 0.0, tension_max_n[free_idx])

		reconstructed = phi @ tension
		residual = wrench_body - reconstructed
		residual_norm = float(np.linalg.norm(residual))
		if residual_norm < best_norm:
			best_tension = tension.copy()
			best_reconstructed = reconstructed.copy()
			best_residual = residual.copy()
			best_norm = residual_norm

	return best_tension, best_reconstructed, best_residual, best_norm


def solve_tension_feasibility(problem: TensionFeasibilityInput) -> TensionFeasibilityResult:
	"""Solve the inner tension-feasibility problem for one sample."""

	q_body, length_error_m = compute_cable_directions_from_uav_positions(
		payload_position_m=problem.payload_position_m,
		payload_rotation_matrix=problem.payload_rotation_matrix,
		uav_positions_m=problem.uav_positions_m,
		attachments=problem.attachments,
	)

	max_length_error = float(np.max(np.abs(length_error_m)))
	if max_length_error > problem.length_tolerance_m:
		return TensionFeasibilityResult(
			feasible=False,
			strictly_feasible=False,
			relaxed_feasible=False,
			message="cable length mismatch exceeds tolerance",
			q_body=q_body,
			tensions_n=np.zeros(problem.attachments.count, dtype=float),
			reconstructed_wrench_body=np.zeros(6, dtype=float),
			residual_wrench_body=np.asarray(problem.wrench_body, dtype=float),
			residual_norm=float(np.linalg.norm(problem.wrench_body)),
			force_residual_norm=float(np.linalg.norm(np.asarray(problem.wrench_body, dtype=float)[:3])),
			torque_residual_norm=float(np.linalg.norm(np.asarray(problem.wrench_body, dtype=float)[3:])),
			force_relative_residual=float("inf"),
			torque_relative_residual=float("inf"),
			cable_length_error_m=length_error_m,
			max_tension_ratio=0.0,
		)

	phi = problem.attachments.wrench_map_body(q_body)
	tensions_n, reconstructed_wrench_body, residual_wrench_body, residual_norm = _solve_bounded_tension_least_squares(
		phi=phi,
		wrench_body=np.asarray(problem.wrench_body, dtype=float),
		tension_max_n=np.asarray(problem.tension_max_n, dtype=float),
	)
	force_residual_norm = float(np.linalg.norm(residual_wrench_body[:3]))
	torque_residual_norm = float(np.linalg.norm(residual_wrench_body[3:]))
	force_target_norm = float(np.linalg.norm(np.asarray(problem.wrench_body, dtype=float)[:3]))
	torque_target_norm = float(np.linalg.norm(np.asarray(problem.wrench_body, dtype=float)[3:]))
	force_relative_residual = force_residual_norm / max(force_target_norm, problem.force_relative_scale_floor_n)
	torque_relative_residual = torque_residual_norm / max(torque_target_norm, problem.torque_relative_scale_floor_nm)
	max_tension_ratio = float(np.max(tensions_n / np.asarray(problem.tension_max_n, dtype=float)))
	strictly_feasible = bool(residual_norm <= problem.residual_tolerance and np.all(tensions_n >= -1e-9))
	relaxed_feasible = bool(
		np.all(tensions_n >= -1e-9)
		and force_residual_norm <= problem.force_residual_absolute_tolerance_n
		and force_relative_residual <= problem.force_residual_relative_tolerance
		and torque_residual_norm <= problem.torque_residual_absolute_tolerance_nm
		and torque_relative_residual <= problem.torque_residual_relative_tolerance
	)
	feasible = relaxed_feasible
	if strictly_feasible:
		message = "strictly feasible"
	elif relaxed_feasible:
		message = "relaxed feasible"
	else:
		message = "wrench residual exceeds relaxed tolerance or bounded solution unavailable"
	return TensionFeasibilityResult(
		feasible=feasible,
		strictly_feasible=strictly_feasible,
		relaxed_feasible=relaxed_feasible,
		message=message,
		q_body=q_body,
		tensions_n=tensions_n,
		reconstructed_wrench_body=reconstructed_wrench_body,
		residual_wrench_body=residual_wrench_body,
		residual_norm=residual_norm,
		force_residual_norm=force_residual_norm,
		torque_residual_norm=torque_residual_norm,
		force_relative_residual=force_relative_residual,
		torque_relative_residual=torque_relative_residual,
		cable_length_error_m=length_error_m,
		max_tension_ratio=max_tension_ratio,
	)