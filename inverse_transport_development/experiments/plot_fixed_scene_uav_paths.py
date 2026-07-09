# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-16
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-16
# 目的 / Purpose:
#   读取固定场景 corridor_export.mat 中的 doors 和 forbidden，并将当前导出的
#   三机 UAV 轨迹叠加绘制出来，用于直接检查在真实固定场景中的避障有效性。
#   Load doors and forbidden obstacles from the fixed-scene corridor_export.mat
#   and overlay the current exported three-UAV trajectories for direct visual
#   inspection of obstacle-avoidance effectiveness in the real fixed scene.
# 主要输入 / Main Inputs:
#   corridor_export.mat，以及 planner_inference_series.npz 中的参考/规划后三机轨迹。
#   corridor_export.mat and the reference/refined three-UAV trajectories stored
#   in planner_inference_series.npz.
# 主要输出 / Main Outputs:
#   固定场景三维/投影视图，以及对 forbidden 和 door ring wall 的碰撞统计。
#   Fixed-scene 3D/projection views and collision statistics against forbidden
#   prisms and door ring walls.
# ==============================================

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import triangulate


def configure_matplotlib_backend(show_window: bool) -> None:
	if show_window:
		# Use an interactive backend so the user can rotate and zoom the 3D scene.
		matplotlib.use("TkAgg")
	else:
		matplotlib.use("Agg")


plt = None
Poly3DCollection = None
DOOR_RENDER_THICKNESS_M = 0.20


def ensure_plot_modules_loaded() -> None:
	global plt, Poly3DCollection
	if plt is None or Poly3DCollection is None:
		import matplotlib.pyplot as _plt
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection

		plt = _plt
		Poly3DCollection = _Poly3DCollection


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
	sys.path.insert(0, str(PKG_ROOT))

from src.common import build_three_uav_box_attachment_set, rotation_matrix_from_rpy
from src.uav_pose_inference import InflatedObstaclePrismXZ, evaluate_cable_obstacle_collisions


@dataclass(frozen=True)
class PrismObstacle:
	polygon_xz: Polygon
	y_min: float
	y_max: float
	label: str
	kind: str


def _unwrap_mat_sequence(value):
	if value is None:
		return []
	if isinstance(value, np.ndarray):
		if value.shape == ():
			return [value.item()]
		return list(np.asarray(value, dtype=object).reshape(-1))
	return [value]


def _as_polygon(poly_xz: np.ndarray) -> Polygon | None:
	poly_xz = np.asarray(poly_xz, dtype=float)
	if poly_xz.ndim != 2 or poly_xz.shape[1] != 2 or poly_xz.shape[0] < 3:
		return None
	polygon = Polygon(poly_xz)
	if polygon.is_empty or polygon.area <= 1e-12:
		return None
	if not polygon.is_valid:
		polygon = polygon.buffer(0.0)
		if polygon.is_empty:
			return None
	return polygon


def _iter_polygons(geometry) -> list[Polygon]:
	if geometry.is_empty:
		return []
	if isinstance(geometry, Polygon):
		return [geometry]
	if isinstance(geometry, MultiPolygon):
		return list(geometry.geoms)
	return []


def load_fixed_scene_obstacles(mat_path: Path, door_ring_margin_m: float) -> tuple[list[PrismObstacle], list[PrismObstacle], list[PrismObstacle]]:
	mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
	doors_raw = _unwrap_mat_sequence(mat.get("doors"))
	forbidden_raw = _unwrap_mat_sequence(mat.get("forbidden"))
	render_door_obstacles: list[PrismObstacle] = []
	door_obstacles: list[PrismObstacle] = []
	forbidden_obstacles: list[PrismObstacle] = []
	for door_index, door in enumerate(doors_raw):
		poly_xz = getattr(door, "poly_xz", None)
		polygon = _as_polygon(poly_xz)
		if polygon is None:
			continue
		y_min = float(getattr(door, "y_min", 0.0))
		y_max = float(getattr(door, "y_max", 0.0))
		y_center = 0.5 * (y_min + y_max)
		half_render_thickness = 0.5 * DOOR_RENDER_THICKNESS_M
		render_door_obstacles.append(
			PrismObstacle(
				polygon_xz=polygon,
				y_min=y_center - half_render_thickness,
				y_max=y_center + half_render_thickness,
				label=f"door_opening_{door_index}",
				kind="door_opening",
			)
		)
		ring_polygon = polygon.buffer(door_ring_margin_m, join_style="mitre").difference(polygon)
		for piece_index, piece in enumerate(_iter_polygons(ring_polygon)):
			door_obstacles.append(
				PrismObstacle(
					polygon_xz=piece,
					y_min=y_min,
					y_max=y_max,
					label=f"door_ring_{door_index}_{piece_index}",
					kind="door_ring",
				)
			)
	for forbidden_index, forbidden in enumerate(forbidden_raw):
		poly_xz = getattr(forbidden, "poly_xz", None)
		polygon = _as_polygon(poly_xz)
		if polygon is None:
			continue
		forbidden_obstacles.append(
			PrismObstacle(
				polygon_xz=polygon,
				y_min=float(getattr(forbidden, "y_min", 0.0)),
				y_max=float(getattr(forbidden, "y_max", 0.0)),
				label=f"forbidden_{forbidden_index}",
				kind="forbidden",
			)
		)
	return render_door_obstacles, door_obstacles, forbidden_obstacles


def _polygon_loops(polygon: Polygon) -> list[np.ndarray]:
	loops = [np.asarray(polygon.exterior.coords, dtype=float)]
	for interior in polygon.interiors:
		loops.append(np.asarray(interior.coords, dtype=float))
	return loops


def add_prism_collection(axis, obstacle: PrismObstacle, face_color: tuple[float, float, float], alpha: float) -> None:
	vertices: list[list[tuple[float, float, float]]] = []
	for poly in triangulate(obstacle.polygon_xz):
		centroid = poly.representative_point()
		if not obstacle.polygon_xz.covers(centroid):
			continue
		coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
		if coords.shape[0] != 3:
			continue
		vertices.append([(float(x), obstacle.y_min, float(z)) for x, z in coords])
		vertices.append([(float(x), obstacle.y_max, float(z)) for x, z in coords[::-1]])
	for loop in _polygon_loops(obstacle.polygon_xz):
		for idx in range(loop.shape[0] - 1):
			x0, z0 = loop[idx]
			x1, z1 = loop[idx + 1]
			vertices.append(
				[
					(float(x0), obstacle.y_min, float(z0)),
					(float(x1), obstacle.y_min, float(z1)),
					(float(x1), obstacle.y_max, float(z1)),
					(float(x0), obstacle.y_max, float(z0)),
				]
			)
	collection = Poly3DCollection(vertices, facecolors=[face_color], edgecolors="none", alpha=alpha)
	axis.add_collection3d(collection)


def obstacle_collision_count(positions_m: np.ndarray, obstacles: list[PrismObstacle]) -> tuple[int, float]:
	positions_m = np.asarray(positions_m, dtype=float)
	collisions = 0
	min_signed_clearance = float("inf")
	for sample_positions in positions_m:
		for point_xyz in sample_positions:
			point_xz = Point(float(point_xyz[0]), float(point_xyz[2]))
			for obstacle in obstacles:
				y_low = min(obstacle.y_min, obstacle.y_max)
				y_high = max(obstacle.y_min, obstacle.y_max)
				inside_y = y_low <= float(point_xyz[1]) <= y_high
				horizontal_distance = obstacle.polygon_xz.exterior.distance(point_xz)
				inside_xz = obstacle.polygon_xz.covers(point_xz)
				if inside_y and inside_xz:
					collisions += 1
					signed_clearance = -float(horizontal_distance)
				else:
					dy = 0.0
					if float(point_xyz[1]) < y_low:
						dy = y_low - float(point_xyz[1])
					elif float(point_xyz[1]) > y_high:
						dy = float(point_xyz[1]) - y_high
					dxz = 0.0 if inside_xz else obstacle.polygon_xz.distance(point_xz)
					signed_clearance = float(np.hypot(dxz, dy))
				min_signed_clearance = min(min_signed_clearance, signed_clearance)
	if not np.isfinite(min_signed_clearance):
		min_signed_clearance = float("nan")
	return collisions, min_signed_clearance


def obstacle_projection_loops(obstacle: PrismObstacle, axes_pair: tuple[int, int]) -> list[np.ndarray]:
	min_x, min_z, max_x, max_z = obstacle.polygon_xz.bounds
	y_low = min(obstacle.y_min, obstacle.y_max)
	y_high = max(obstacle.y_min, obstacle.y_max)
	if axes_pair == (0, 2):
		return [np.asarray(obstacle.polygon_xz.exterior.coords, dtype=float)]
	if axes_pair == (0, 1):
		return [np.array([[min_x, y_low], [max_x, y_low], [max_x, y_high], [min_x, y_high], [min_x, y_low]], dtype=float)]
	if axes_pair == (2, 1):
		return [np.array([[min_z, y_low], [max_z, y_low], [max_z, y_high], [min_z, y_high], [min_z, y_low]], dtype=float)]
	raise ValueError(f"unsupported axes_pair: {axes_pair}")


def save_projection_plot(output_path: Path, payload_positions_m: np.ndarray, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, render_door_obstacles: list[PrismObstacle], door_obstacles: list[PrismObstacle], forbidden_obstacles: list[PrismObstacle], axes_pair: tuple[int, int], axis_labels: tuple[str, str], title: str) -> None:
	ensure_plot_modules_loaded()
	fig, axis = plt.subplots(figsize=(8.5, 7.5))
	axis.plot(payload_positions_m[:, axes_pair[0]], payload_positions_m[:, axes_pair[1]], color="black", linewidth=2.0, label="payload")
	for obstacle in render_door_obstacles:
		for loop in obstacle_projection_loops(obstacle, axes_pair):
			axis.fill(loop[:, 0], loop[:, 1], color="#8ec07c", alpha=0.18)
	for obstacle in forbidden_obstacles:
		for loop in obstacle_projection_loops(obstacle, axes_pair):
			axis.fill(loop[:, 0], loop[:, 1], color="tab:red", alpha=0.18)
	for obstacle in door_obstacles:
		for loop in obstacle_projection_loops(obstacle, axes_pair):
			axis.fill(loop[:, 0], loop[:, 1], color="tab:blue", alpha=0.08)
	for uav_index in range(reference_positions_m.shape[1]):
		axis.plot(reference_positions_m[:, uav_index, axes_pair[0]], reference_positions_m[:, uav_index, axes_pair[1]], linestyle="--", alpha=0.65, label=f"ref_uav{uav_index+1}")
		axis.plot(refined_positions_m[:, uav_index, axes_pair[0]], refined_positions_m[:, uav_index, axes_pair[1]], linewidth=2.0, label=f"plan_uav{uav_index+1}")
	axis.set_xlabel(axis_labels[0])
	axis.set_ylabel(axis_labels[1])
	axis.set_title(title)
	axis.grid(True, alpha=0.3)
	axis.axis("equal")
	axis.legend(loc="best", ncol=2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def build_3d_figure(payload_positions_m: np.ndarray, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, render_door_obstacles: list[PrismObstacle], door_obstacles: list[PrismObstacle], forbidden_obstacles: list[PrismObstacle]):
	ensure_plot_modules_loaded()
	fig = plt.figure(figsize=(10, 8))
	axis = fig.add_subplot(111, projection="3d")
	for obstacle in render_door_obstacles:
		add_prism_collection(axis, obstacle, face_color=(0.56, 0.77, 0.49), alpha=0.22)
	for obstacle in door_obstacles:
		add_prism_collection(axis, obstacle, face_color=(0.63, 0.76, 0.92), alpha=0.10)
	for obstacle in forbidden_obstacles:
		add_prism_collection(axis, obstacle, face_color=(0.25, 0.25, 0.25), alpha=0.18)
	axis.plot(payload_positions_m[:, 0], payload_positions_m[:, 1], payload_positions_m[:, 2], color="black", linewidth=2.4, label="payload")
	for uav_index in range(reference_positions_m.shape[1]):
		axis.plot(reference_positions_m[:, uav_index, 0], reference_positions_m[:, uav_index, 1], reference_positions_m[:, uav_index, 2], linestyle="--", alpha=0.35, linewidth=1.2, label=f"ref_uav{uav_index+1}")
		axis.plot(refined_positions_m[:, uav_index, 0], refined_positions_m[:, uav_index, 1], refined_positions_m[:, uav_index, 2], linewidth=2.4, label=f"plan_uav{uav_index+1}")
		axis.scatter(refined_positions_m[0, uav_index, 0], refined_positions_m[0, uav_index, 1], refined_positions_m[0, uav_index, 2], s=36, marker="o")
		axis.scatter(refined_positions_m[-1, uav_index, 0], refined_positions_m[-1, uav_index, 1], refined_positions_m[-1, uav_index, 2], s=42, marker="^")
	axis.scatter(payload_positions_m[0, 0], payload_positions_m[0, 1], payload_positions_m[0, 2], color="black", s=42, marker="o")
	axis.scatter(payload_positions_m[-1, 0], payload_positions_m[-1, 1], payload_positions_m[-1, 2], color="black", s=52, marker="^")
	axis.set_xlabel("x [m]")
	axis.set_ylabel("y [m]")
	axis.set_zlabel("z [m]")
	axis.set_title("Fixed Scene: Doors + Forbidden + UAV Paths")
	all_positions_m = np.concatenate(
		[
			payload_positions_m,
			reference_positions_m.reshape(-1, 3),
			refined_positions_m.reshape(-1, 3),
		],
		axis=0,
	)
	x_span = max(float(np.ptp(all_positions_m[:, 0])), 1e-6)
	y_span = max(float(np.ptp(all_positions_m[:, 1])), 1e-6)
	z_span = max(float(np.ptp(all_positions_m[:, 2])), 1e-6)
	axis.set_box_aspect((x_span, 1.45 * y_span, z_span))
	axis.view_init(elev=24.0, azim=-56.0)
	axis.legend(loc="best", ncol=2)
	fig.tight_layout()
	return fig


def save_3d_plot(output_path: Path, payload_positions_m: np.ndarray, reference_positions_m: np.ndarray, refined_positions_m: np.ndarray, render_door_obstacles: list[PrismObstacle], door_obstacles: list[PrismObstacle], forbidden_obstacles: list[PrismObstacle]) -> None:
	fig = build_3d_figure(
		payload_positions_m=payload_positions_m,
		reference_positions_m=reference_positions_m,
		refined_positions_m=refined_positions_m,
		render_door_obstacles=render_door_obstacles,
		door_obstacles=door_obstacles,
		forbidden_obstacles=forbidden_obstacles,
	)
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot three-UAV paths inside the fixed scene with doors and forbidden prisms")
	parser.add_argument("--mat", type=Path, default=Path("corridor_export.mat"), help="Path to fixed-scene corridor_export.mat")
	parser.add_argument(
		"--npz",
		type=Path,
		default=Path("inverse_transport_development/results/planner_export_inference/planner_inference_series.npz"),
		help="Path to planner_inference_series.npz",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("inverse_transport_development/results/fixed_scene_uav_plots"),
		help="Directory used to save fixed-scene UAV plots",
	)
	parser.add_argument("--door-ring-margin", type=float, default=0.55, help="Door wall ring margin used by the fixed-scene visualization")
	parser.add_argument("--show", action="store_true", help="Open an interactive 3D plot window so the scene can be rotated manually")
	parser.add_argument("--box-size", type=float, nargs=3, default=[0.8, 0.3, 0.2], help="Payload box size used to reconstruct cable anchor points")
	args = parser.parse_args()
	configure_matplotlib_backend(show_window=bool(args.show))
	ensure_plot_modules_loaded()

	render_door_obstacles, door_obstacles, forbidden_obstacles = load_fixed_scene_obstacles(args.mat, door_ring_margin_m=float(args.door_ring_margin))
	bundle = np.load(args.npz)
	payload_positions_m = np.asarray(bundle["payload_position_m"], dtype=float)
	payload_orientation_rpy_rad = np.asarray(bundle["payload_orientation_rpy_rad"], dtype=float)
	reference_positions_m = np.asarray(bundle["quadrotor_position_reference_inertial_m"], dtype=float)
	refined_positions_m = np.asarray(bundle["quadrotor_position_inertial_m"], dtype=float)
	attachments = build_three_uav_box_attachment_set(np.asarray(args.box_size, dtype=float))
	payload_rotation_matrices = np.asarray([rotation_matrix_from_rpy(rpy) for rpy in payload_orientation_rpy_rad], dtype=float)
	anchor_positions_m = np.zeros((payload_positions_m.shape[0], attachments.count, 3), dtype=float)
	for sample_index in range(payload_positions_m.shape[0]):
		anchor_positions_m[sample_index] = payload_positions_m[sample_index][None, :] + (payload_rotation_matrices[sample_index] @ attachments.r_i_body_m.T).T
	args.output_dir.mkdir(parents=True, exist_ok=True)
	save_projection_plot(
		args.output_dir / "fixed_scene_uav_xy.png",
		payload_positions_m=payload_positions_m,
		reference_positions_m=reference_positions_m,
		refined_positions_m=refined_positions_m,
		render_door_obstacles=render_door_obstacles,
		door_obstacles=door_obstacles,
		forbidden_obstacles=forbidden_obstacles,
		axes_pair=(0, 1),
		axis_labels=("x [m]", "y [m]"),
		title="Fixed Scene XY Projection",
	)
	save_projection_plot(
		args.output_dir / "fixed_scene_uav_xz.png",
		payload_positions_m=payload_positions_m,
		reference_positions_m=reference_positions_m,
		refined_positions_m=refined_positions_m,
		render_door_obstacles=render_door_obstacles,
		door_obstacles=door_obstacles,
		forbidden_obstacles=forbidden_obstacles,
		axes_pair=(0, 2),
		axis_labels=("x [m]", "z [m]"),
		title="Fixed Scene XZ Projection",
	)
	save_3d_plot(
		args.output_dir / "fixed_scene_uav_3d.png",
		payload_positions_m=payload_positions_m,
		reference_positions_m=reference_positions_m,
		refined_positions_m=refined_positions_m,
		render_door_obstacles=render_door_obstacles,
		door_obstacles=door_obstacles,
		forbidden_obstacles=forbidden_obstacles,
	)
	ref_forbidden_collisions, ref_forbidden_clearance = obstacle_collision_count(reference_positions_m, forbidden_obstacles)
	plan_forbidden_collisions, plan_forbidden_clearance = obstacle_collision_count(refined_positions_m, forbidden_obstacles)
	ref_door_collisions, ref_door_clearance = obstacle_collision_count(reference_positions_m, door_obstacles)
	plan_door_collisions, plan_door_clearance = obstacle_collision_count(refined_positions_m, door_obstacles)
	all_obstacles = [*door_obstacles, *forbidden_obstacles]
	all_scene_obstacles = [
		InflatedObstaclePrismXZ(
			polygon_xz=np.asarray(obstacle.polygon_xz.exterior.coords[:-1], dtype=float),
			y_min_m=float(obstacle.y_min),
			y_max_m=float(obstacle.y_max),
			label=obstacle.label,
		)
		for obstacle in all_obstacles
	]
	ref_cable_collisions, ref_cable_clearance = evaluate_cable_obstacle_collisions(anchor_positions_m, reference_positions_m, all_scene_obstacles)
	plan_cable_collisions, plan_cable_clearance = evaluate_cable_obstacle_collisions(anchor_positions_m, refined_positions_m, all_scene_obstacles)
	print(f"[fixed_scene_plot] doors={len(door_obstacles)} forbidden={len(forbidden_obstacles)}")
	print(f"[fixed_scene_plot] reference_forbidden_collisions={ref_forbidden_collisions} min_clearance={ref_forbidden_clearance:.6f} m")
	print(f"[fixed_scene_plot] refined_forbidden_collisions={plan_forbidden_collisions} min_clearance={plan_forbidden_clearance:.6f} m")
	print(f"[fixed_scene_plot] reference_door_wall_collisions={ref_door_collisions} min_clearance={ref_door_clearance:.6f} m")
	print(f"[fixed_scene_plot] refined_door_wall_collisions={plan_door_collisions} min_clearance={plan_door_clearance:.6f} m")
	print(f"[fixed_scene_plot] reference_cable_collisions={ref_cable_collisions} min_clearance={ref_cable_clearance:.6f} m")
	print(f"[fixed_scene_plot] refined_cable_collisions={plan_cable_collisions} min_clearance={plan_cable_clearance:.6f} m")
	print(f"[fixed_scene_plot] saved {args.output_dir / 'fixed_scene_uav_xy.png'}")
	print(f"[fixed_scene_plot] saved {args.output_dir / 'fixed_scene_uav_xz.png'}")
	print(f"[fixed_scene_plot] saved {args.output_dir / 'fixed_scene_uav_3d.png'}")
	if args.show:
		interactive_fig = build_3d_figure(
			payload_positions_m=payload_positions_m,
			reference_positions_m=reference_positions_m,
			refined_positions_m=refined_positions_m,
			render_door_obstacles=render_door_obstacles,
			door_obstacles=door_obstacles,
			forbidden_obstacles=forbidden_obstacles,
		)
		print("[fixed_scene_plot] opening interactive 3D window")
		plt.show()
		plt.close(interactive_fig)


if __name__ == "__main__":
	main()