# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-10
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-10
# 目的 / Purpose:
#   定义 UAV 安全走廊的基础数据结构，并提供可立即使用的参考轨迹局部盒走廊
#   生成函数，作为后续多面体 corridor 构造的最小工程骨架。
#   Define core safe-corridor data structures and provide a usable local box
#   corridor builder around a reference trajectory as the minimum engineering
#   scaffold for later polyhedral corridor generation.
# 主要输入 / Main Inputs:
#   参考 UAV 轨迹、局部走廊宽度、可选膨胀障碍物占位信息。
#   Reference UAV trajectories, local corridor widths, and optional inflated
#   obstacle placeholders.
# 主要输出 / Main Outputs:
#   每个 UAV、每个时刻的半空间走廊表示 A x <= b。
#   Per-UAV, per-sample halfspace corridors of the form A x <= b.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def _validate_points(name: str, value: np.ndarray) -> np.ndarray:
	array = np.asarray(value, dtype=float)
	if array.ndim != 3 or array.shape[2] != 3:
		raise ValueError(f"{name} must have shape (N, M, 3)")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be finite")
	return array


def _validate_vector3(name: str, value: np.ndarray, positive: bool = False) -> np.ndarray:
	array = np.asarray(value, dtype=float).reshape(-1)
	if array.shape != (3,) or not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be a finite 3-vector")
	if positive and np.any(array <= 0.0):
		raise ValueError(f"{name} must be strictly positive")
	return array


def _validate_polygon_xz(name: str, value: np.ndarray) -> np.ndarray:
	array = np.asarray(value, dtype=float)
	if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
		raise ValueError(f"{name} must have shape (K, 2) with K >= 3")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be finite")
	return array


def _to_shapely_polygon(polygon_xz: np.ndarray) -> Polygon:
	polygon = Polygon(np.asarray(polygon_xz, dtype=float))
	if not polygon.is_valid:
		polygon = polygon.buffer(0.0)
	if polygon.is_empty or polygon.area <= 1e-12:
		raise ValueError("polygon_xz must define a non-empty area")
	if not isinstance(polygon, Polygon):
		raise ValueError("polygon_xz must resolve to a single polygon")
	return polygon


@dataclass(frozen=True)
class InflatedObstacleSphere:
	"""Conservative spherical placeholder for an inflated obstacle."""

	center_m: np.ndarray
	radius_m: float
	label: str = "unknown"

	def __post_init__(self) -> None:
		center_m = np.asarray(self.center_m, dtype=float).reshape(-1)
		if center_m.shape != (3,) or not np.all(np.isfinite(center_m)):
			raise ValueError("center_m must be a finite 3-vector")
		if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
			raise ValueError("radius_m must be finite and positive")


@dataclass(frozen=True)
class InflatedObstaclePrismXZ:
	"""Extruded obstacle with an x-z polygon footprint and y bounds."""

	polygon_xz: np.ndarray
	y_min_m: float
	y_max_m: float
	label: str = "unknown"

	def __post_init__(self) -> None:
		polygon_xz = _validate_polygon_xz("polygon_xz", self.polygon_xz)
		_to_shapely_polygon(polygon_xz)
		if not np.isfinite(self.y_min_m) or not np.isfinite(self.y_max_m):
			raise ValueError("y_min_m and y_max_m must be finite")
		if abs(float(self.y_max_m) - float(self.y_min_m)) <= 1e-9:
			raise ValueError("y_min_m and y_max_m must define a non-zero height")


SceneObstacle = InflatedObstacleSphere | InflatedObstaclePrismXZ


@dataclass(frozen=True)
class HalfspaceCorridor3D:
	"""Polyhedral corridor of the form A x <= b for a single UAV sample."""

	A: np.ndarray
	b: np.ndarray
	reference_point_m: np.ndarray
	uav_index: int
	sample_index: int

	def __post_init__(self) -> None:
		A = np.asarray(self.A, dtype=float)
		b = np.asarray(self.b, dtype=float).reshape(-1)
		reference_point_m = np.asarray(self.reference_point_m, dtype=float).reshape(-1)
		if A.ndim != 2 or A.shape[1] != 3:
			raise ValueError("A must have shape (m, 3)")
		if b.shape != (A.shape[0],):
			raise ValueError("b must align with A")
		if reference_point_m.shape != (3,) or not np.all(np.isfinite(reference_point_m)):
			raise ValueError("reference_point_m must be a finite 3-vector")
		if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
			raise ValueError("A and b must be finite")

	def contains(self, point_m: np.ndarray, tolerance: float = 1e-9) -> bool:
		point_m = np.asarray(point_m, dtype=float).reshape(-1)
		if point_m.shape != (3,):
			raise ValueError("point_m must be a 3-vector")
		return bool(np.all(self.A @ point_m <= self.b + tolerance))

	def max_violation(self, point_m: np.ndarray) -> float:
		point_m = np.asarray(point_m, dtype=float).reshape(-1)
		if point_m.shape != (3,):
			raise ValueError("point_m must be a 3-vector")
		return float(np.max(self.A @ point_m - self.b))


@dataclass(frozen=True)
class CorridorGenerationConfig:
	"""Configuration for generating local safe corridors from a reference path."""

	half_width_xyz_m: np.ndarray
	tracking_margin_m: float = 0.0

	def __post_init__(self) -> None:
		half_width_xyz_m = _validate_vector3("half_width_xyz_m", self.half_width_xyz_m, positive=True)
		if not np.isfinite(self.tracking_margin_m) or self.tracking_margin_m < 0.0:
			raise ValueError("tracking_margin_m must be finite and non-negative")

	@property
	def effective_half_width_xyz_m(self) -> np.ndarray:
		return np.asarray(self.half_width_xyz_m, dtype=float) + float(self.tracking_margin_m)


def make_box_halfspaces(center_m: np.ndarray, half_width_xyz_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Build axis-aligned box halfspaces A x <= b around a center point."""

	center_m = _validate_vector3("center_m", center_m)
	half_width_xyz_m = _validate_vector3("half_width_xyz_m", half_width_xyz_m, positive=True)

	A = np.array(
		[
			[1.0, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, -1.0, 0.0],
			[0.0, 0.0, 1.0],
			[0.0, 0.0, -1.0],
		],
		dtype=float,
	)
	b = np.array(
		[
			center_m[0] + half_width_xyz_m[0],
			-center_m[0] + half_width_xyz_m[0],
			center_m[1] + half_width_xyz_m[1],
			-center_m[1] + half_width_xyz_m[1],
			center_m[2] + half_width_xyz_m[2],
			-center_m[2] + half_width_xyz_m[2],
		],
		dtype=float,
	)
	return A, b


def make_sphere_exclusion_halfspace(
	reference_point_m: np.ndarray,
	obstacle: InflatedObstacleSphere,
	eps: float = 1e-9,
) -> tuple[np.ndarray, float] | None:
	"""Build a supporting halfspace that excludes an inflated sphere and keeps the reference point.

	When the reference point lies outside the sphere, the halfspace is tangent to the
	sphere and aligned with the radial direction from the obstacle center to the
	reference point. If the reference point lies inside the inflated sphere, no valid
	separating tangent plane exists, so this function returns None.
	"""

	reference_point_m = _validate_vector3("reference_point_m", reference_point_m)
	center_m = np.asarray(obstacle.center_m, dtype=float).reshape(-1)
	radius_m = float(obstacle.radius_m)
	delta = reference_point_m - center_m
	distance = float(np.linalg.norm(delta))
	if distance <= radius_m + eps:
		return None
	normal = delta / max(distance, eps)
	# Enforce normal^T x >= normal^T c + r, rewritten as (-normal)^T x <= -(normal^T c + r).
	a_row = -normal
	b_value = -(float(normal @ center_m) + radius_m)
	return a_row, b_value


def make_prism_exclusion_halfspace(
	reference_point_m: np.ndarray,
	obstacle: InflatedObstaclePrismXZ,
	eps: float = 1e-9,
) -> tuple[np.ndarray, float] | None:
	"""Build a local separating halfspace for an extruded x-z prism obstacle.

	For points outside the prism, the supporting plane is constructed from the
	nearest point on the prism. For points inside the prism, the plane is placed
	at the nearest exit surface so the local QP can push the sample out of the
	obstacle instead of silently dropping the constraint.
	"""

	reference_point_m = _validate_vector3("reference_point_m", reference_point_m)
	polygon = _to_shapely_polygon(obstacle.polygon_xz)
	point_xz = Point(float(reference_point_m[0]), float(reference_point_m[2]))
	y_low = min(float(obstacle.y_min_m), float(obstacle.y_max_m))
	y_high = max(float(obstacle.y_min_m), float(obstacle.y_max_m))
	point_y = float(reference_point_m[1])
	inside_xz = bool(polygon.covers(point_xz))
	inside_y = y_low <= point_y <= y_high

	nearest_xz = np.array([reference_point_m[0], reference_point_m[2]], dtype=float)
	if not inside_xz:
		_, nearest_boundary_point = nearest_points(point_xz, polygon)
		nearest_xz = np.array([nearest_boundary_point.x, nearest_boundary_point.y], dtype=float)
	nearest_y = min(max(point_y, y_low), y_high)
	nearest_point = np.array([nearest_xz[0], nearest_y, nearest_xz[1]], dtype=float)
	outside_delta = reference_point_m - nearest_point
	outside_distance = float(np.linalg.norm(outside_delta))
	if outside_distance > eps:
		normal = outside_delta / outside_distance
		a_row = -normal
		b_value = -float(normal @ nearest_point)
		return a_row, b_value

	if not (inside_xz and inside_y):
		return None

	_, nearest_boundary_point = nearest_points(point_xz, polygon.exterior)
	boundary_xz = np.array([nearest_boundary_point.x, nearest_boundary_point.y], dtype=float)
	side_distance = float(np.linalg.norm(boundary_xz - reference_point_m[[0, 2]]))
	bottom_distance = abs(point_y - y_low)
	top_distance = abs(y_high - point_y)
	min_distance = min(side_distance, bottom_distance, top_distance)
	if min_distance <= eps:
		return None
	if min_distance == side_distance:
		normal_xz = boundary_xz - reference_point_m[[0, 2]]
		normal_xz_norm = float(np.linalg.norm(normal_xz))
		if normal_xz_norm <= eps:
			return None
		normal = np.array([normal_xz[0] / normal_xz_norm, 0.0, normal_xz[1] / normal_xz_norm], dtype=float)
		boundary_point = np.array([boundary_xz[0], point_y, boundary_xz[1]], dtype=float)
	elif min_distance == bottom_distance:
		normal = np.array([0.0, -1.0, 0.0], dtype=float)
		boundary_point = np.array([reference_point_m[0], y_low, reference_point_m[2]], dtype=float)
	else:
		normal = np.array([0.0, 1.0, 0.0], dtype=float)
		boundary_point = np.array([reference_point_m[0], y_high, reference_point_m[2]], dtype=float)
	a_row = -normal
	b_value = -float(normal @ boundary_point)
	return a_row, b_value


def make_obstacle_exclusion_halfspace(
	reference_point_m: np.ndarray,
	obstacle: SceneObstacle,
	eps: float = 1e-9,
) -> tuple[np.ndarray, float] | None:
	if isinstance(obstacle, InflatedObstacleSphere):
		return make_sphere_exclusion_halfspace(reference_point_m, obstacle, eps=eps)
	if isinstance(obstacle, InflatedObstaclePrismXZ):
		return make_prism_exclusion_halfspace(reference_point_m, obstacle, eps=eps)
	raise TypeError("unsupported obstacle type")


def signed_distance_to_obstacle(point_m: np.ndarray, obstacle: SceneObstacle) -> float:
	"""Return a signed clearance: positive outside, negative inside, zero on boundary."""

	point_m = _validate_vector3("point_m", point_m)
	if isinstance(obstacle, InflatedObstacleSphere):
		center_m = np.asarray(obstacle.center_m, dtype=float).reshape(-1)
		return float(np.linalg.norm(point_m - center_m) - float(obstacle.radius_m))
	if isinstance(obstacle, InflatedObstaclePrismXZ):
		polygon = _to_shapely_polygon(obstacle.polygon_xz)
		point_xz = Point(float(point_m[0]), float(point_m[2]))
		y_low = min(float(obstacle.y_min_m), float(obstacle.y_max_m))
		y_high = max(float(obstacle.y_min_m), float(obstacle.y_max_m))
		inside_xz = bool(polygon.covers(point_xz))
		inside_y = y_low <= float(point_m[1]) <= y_high
		if inside_xz and inside_y:
			horizontal_to_boundary = float(polygon.exterior.distance(point_xz))
			vertical_to_boundary = min(float(point_m[1]) - y_low, y_high - float(point_m[1]))
			return -min(horizontal_to_boundary, vertical_to_boundary)
		dy = 0.0
		if float(point_m[1]) < y_low:
			dy = y_low - float(point_m[1])
		elif float(point_m[1]) > y_high:
			dy = float(point_m[1]) - y_high
		dxz = 0.0 if inside_xz else float(polygon.distance(point_xz))
		return float(np.hypot(dxz, dy))
	raise TypeError("unsupported obstacle type")


def evaluate_obstacle_collisions(positions_m: np.ndarray, obstacles: list[SceneObstacle] | tuple[SceneObstacle, ...]) -> tuple[int, float]:
	"""Count per-sample obstacle penetrations and track the minimum signed clearance."""

	positions_m = _validate_points("positions_m", positions_m)
	collision_count = 0
	min_signed_clearance_m = float("inf")
	for sample_positions in positions_m:
		for point_m in sample_positions:
			for obstacle in obstacles:
				signed_clearance_m = signed_distance_to_obstacle(point_m, obstacle)
				if signed_clearance_m < 0.0:
					collision_count += 1
				min_signed_clearance_m = min(min_signed_clearance_m, signed_clearance_m)
	if not np.isfinite(min_signed_clearance_m):
		min_signed_clearance_m = float("nan")
	return collision_count, min_signed_clearance_m


def sample_segment_points(start_point_m: np.ndarray, end_point_m: np.ndarray, n_samples: int = 9) -> np.ndarray:
	"""Uniformly sample a 3D line segment, including both endpoints."""

	start_point_m = _validate_vector3("start_point_m", start_point_m)
	end_point_m = _validate_vector3("end_point_m", end_point_m)
	if not isinstance(n_samples, int) or n_samples < 2:
		raise ValueError("n_samples must be an integer >= 2")
	alphas = np.linspace(0.0, 1.0, n_samples, dtype=float)
	return start_point_m[None, :] + alphas[:, None] * (end_point_m - start_point_m)[None, :]


def evaluate_cable_obstacle_collisions(
	anchor_positions_m: np.ndarray,
	uav_positions_m: np.ndarray,
	obstacles: list[SceneObstacle] | tuple[SceneObstacle, ...],
	n_samples_per_cable: int = 9,
) -> tuple[int, float]:
	"""Approximate cable collision count by sampling each payload-anchor-to-UAV segment."""

	anchor_positions_m = np.asarray(anchor_positions_m, dtype=float)
	uav_positions_m = np.asarray(uav_positions_m, dtype=float)
	if anchor_positions_m.shape != uav_positions_m.shape or anchor_positions_m.ndim != 3 or anchor_positions_m.shape[2] != 3:
		raise ValueError("anchor_positions_m and uav_positions_m must both have shape (N, M, 3)")
	collision_count = 0
	min_signed_clearance_m = float("inf")
	for sample_index in range(anchor_positions_m.shape[0]):
		for uav_index in range(anchor_positions_m.shape[1]):
			segment_points_m = sample_segment_points(
				anchor_positions_m[sample_index, uav_index],
				uav_positions_m[sample_index, uav_index],
				n_samples=n_samples_per_cable,
			)
			segment_collision = False
			for obstacle in obstacles:
				obstacle_min_clearance_m = float("inf")
				for point_m in segment_points_m:
					signed_clearance_m = signed_distance_to_obstacle(point_m, obstacle)
					obstacle_min_clearance_m = min(obstacle_min_clearance_m, signed_clearance_m)
					min_signed_clearance_m = min(min_signed_clearance_m, signed_clearance_m)
				if obstacle_min_clearance_m < 0.0:
					segment_collision = True
			if segment_collision:
				collision_count += 1
	if not np.isfinite(min_signed_clearance_m):
		min_signed_clearance_m = float("nan")
	return collision_count, min_signed_clearance_m


def build_axis_aligned_corridors_from_reference(
	reference_positions_m: np.ndarray,
	config: CorridorGenerationConfig,
	obstacles: list[SceneObstacle] | None = None,
) -> list[list[HalfspaceCorridor3D]]:
	"""Build per-sample local box corridors around a reference UAV trajectory.

	The first scaffold intentionally keeps the corridor generator conservative and
	free of heavy geometry dependencies. Obstacle inputs are accepted so the API
	shape is stable, but obstacle carving is deferred to the next iteration.
	"""

	reference_positions_m = _validate_points("reference_positions_m", reference_positions_m)
	if obstacles is not None:
		for obstacle in obstacles:
			if not isinstance(obstacle, (InflatedObstacleSphere, InflatedObstaclePrismXZ)):
				raise TypeError("obstacles must contain supported obstacle instances")

	half_width_xyz_m = config.effective_half_width_xyz_m
	n_sample, n_uav, _ = reference_positions_m.shape
	result: list[list[HalfspaceCorridor3D]] = []
	for sample_index in range(n_sample):
		sample_corridors: list[HalfspaceCorridor3D] = []
		for uav_index in range(n_uav):
			reference_point = reference_positions_m[sample_index, uav_index]
			A, b = make_box_halfspaces(reference_point, half_width_xyz_m=half_width_xyz_m)
			if obstacles:
				rows = [A]
				bounds = [b]
				for obstacle in obstacles:
					halfspace = make_obstacle_exclusion_halfspace(reference_point, obstacle)
					if halfspace is None:
						continue
					a_row, b_value = halfspace
					box_support = float(np.sum(np.abs(a_row) * half_width_xyz_m))
					if float(a_row @ reference_point) + box_support <= float(b_value) + 1e-9:
						continue
					rows.append(a_row.reshape(1, 3))
					bounds.append(np.array([b_value], dtype=float))
				A = np.vstack(rows)
				b = np.concatenate(bounds)
			sample_corridors.append(
				HalfspaceCorridor3D(
					A=A,
					b=b,
					reference_point_m=reference_point,
					uav_index=uav_index,
					sample_index=sample_index,
				)
			)
		result.append(sample_corridors)
	return result