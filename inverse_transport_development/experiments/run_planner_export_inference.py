# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-26
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-26
# 目的 / Purpose:
#   读取当前规划阶段导出的真实轨迹文件（优先 corridor_export.mat），将其接入
#   payload wrench、挂点力分配与 UAV 位置反推链路，并输出折线图与中间结果。
#   Load real planner-exported trajectory files (prefer corridor_export.mat),
#   feed them into the payload wrench / attachment-force / UAV-position
#   inference pipeline, and export plots plus intermediate results.
# 主要输入 / Main Inputs:
#   corridor_export.mat 或 sample_t/sample_xyz npy 文件，载荷质量与长方体尺寸。
#   corridor_export.mat or sample_t/sample_xyz npy files, payload mass, and box
#   size.
# 主要输出 / Main Outputs:
#   载荷轨迹、wrench、挂点力、无人机位置折线图，以及 npz 中间结果。
#   Trajectory/wrench/attachment-force/UAV-position plots and an npz bundle.
# ==============================================

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.cable_force_inference import PayloadGeometry, PayloadPhysicalParams, compute_payload_wrench, infer_cable_force_series
from src.common import build_three_uav_box_attachment_set, rotation_matrix_from_rpy
from src.common.trajectory import TrajectorySamples3D
from src.uav_pose_inference import CorridorGenerationConfig, InflatedObstaclePrismXZ, SequentialPlanningConfig, UAVKinematicLimits, evaluate_cable_obstacle_collisions, plan_collision_aware_uav_positions


def _unwrap_mat_scalar(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _extract_roll_from_mat(mat: dict, sample_t: np.ndarray) -> np.ndarray:
    if "keyframes" not in mat or "traj" not in mat:
        return np.zeros_like(sample_t)
    keyframes = mat["keyframes"]
    traj = mat["traj"]
    try:
        roll_wp = np.asarray(keyframes[0][0]["roll_wp"], dtype=float).reshape(-1)
        t_per_seg = np.asarray(traj[0][0]["T_per_seg"], dtype=float).reshape(-1)
    except Exception:
        return np.zeros_like(sample_t)
    if roll_wp.size == 0 or t_per_seg.size == 0:
        return np.zeros_like(sample_t)
    wp_t = np.zeros(t_per_seg.size + 1, dtype=float)
    wp_t[1:] = np.cumsum(t_per_seg)
    return np.interp(sample_t, wp_t, roll_wp)


def _deduplicate_time_aligned_series(sample_t: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop repeated timestamps while preserving the first occurrence order."""

    sample_t = np.asarray(sample_t, dtype=float).reshape(-1)
    _, unique_idx = np.unique(np.round(sample_t, decimals=12), return_index=True)
    unique_idx = np.sort(unique_idx)
    outputs = [sample_t[unique_idx]]
    for array in arrays:
        outputs.append(np.asarray(array)[unique_idx])
    return tuple(outputs)


def _unwrap_mat_sequence(value) -> list[object]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return [value.item()]
        return list(np.asarray(value, dtype=object).reshape(-1))
    return [value]


def _as_single_polygon(poly_xz: np.ndarray) -> Polygon | None:
    poly_xz = np.asarray(poly_xz, dtype=float)
    if poly_xz.ndim != 2 or poly_xz.shape[1] != 2 or poly_xz.shape[0] < 3:
        return None
    polygon = Polygon(poly_xz)
    if not polygon.is_valid:
        polygon = polygon.buffer(0.0)
    if polygon.is_empty or polygon.area <= 1e-12 or not isinstance(polygon, Polygon):
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


def _triangulate_polygon_region(polygon: Polygon) -> list[np.ndarray]:
    triangles: list[np.ndarray] = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle.representative_point()):
            continue
        coords = np.asarray(triangle.exterior.coords[:-1], dtype=float)
        if coords.shape[0] < 3:
            continue
        triangles.append(coords)
    return triangles


def build_fixed_scene_obstacles(mat_path: Path, door_ring_margin_m: float) -> list[InflatedObstaclePrismXZ]:
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    obstacles: list[InflatedObstaclePrismXZ] = []
    for door_index, door in enumerate(_unwrap_mat_sequence(mat.get("doors"))):
        polygon = _as_single_polygon(getattr(door, "poly_xz", None))
        if polygon is None:
            continue
        ring_polygon = polygon.buffer(door_ring_margin_m, join_style="mitre").difference(polygon)
        y_min_m = float(getattr(door, "y_min", 0.0))
        y_max_m = float(getattr(door, "y_max", 0.0))
        for piece_index, piece in enumerate(_iter_polygons(ring_polygon)):
            for tri_index, triangle_coords in enumerate(_triangulate_polygon_region(piece)):
                obstacles.append(
                    InflatedObstaclePrismXZ(
                        polygon_xz=triangle_coords,
                        y_min_m=y_min_m,
                        y_max_m=y_max_m,
                        label=f"door_ring_{door_index}_{piece_index}_tri_{tri_index}",
                    )
                )
    for forbidden_index, forbidden in enumerate(_unwrap_mat_sequence(mat.get("forbidden"))):
        polygon = _as_single_polygon(getattr(forbidden, "poly_xz", None))
        if polygon is None:
            continue
        obstacles.append(
            InflatedObstaclePrismXZ(
                polygon_xz=np.asarray(polygon.exterior.coords[:-1], dtype=float),
                y_min_m=float(getattr(forbidden, "y_min", 0.0)),
                y_max_m=float(getattr(forbidden, "y_max", 0.0)),
                label=f"forbidden_{forbidden_index}",
            )
        )
    return obstacles


def load_passage_hints(mat_path: Path | None) -> list[dict[str, float | int | str]]:
    if mat_path is None or not mat_path.exists():
        return []
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    hints = []
    for hint in _unwrap_mat_sequence(mat.get("uav_passage_hints")):
        kind = getattr(hint, "kind", None)
        if kind is None:
            continue
        hints.append(
            {
                "kind": str(kind),
                "y_start": float(getattr(hint, "y_start", 0.0)),
                "y_end": float(getattr(hint, "y_end", 0.0)),
                "y_center": float(getattr(hint, "y_center", 0.0)),
                "leader_index": int(getattr(hint, "leader_index", 0)),
                "forward_offset_m": float(getattr(hint, "forward_offset_m", 0.0)),
                "holdback_offset_m": float(getattr(hint, "holdback_offset_m", 0.0)),
                "flatten_scale": float(getattr(hint, "flatten_scale", 1.0)),
                "leader_pre_scale": float(getattr(hint, "leader_pre_scale", 1.0)),
                "follower_pre_holdback_scale": float(getattr(hint, "follower_pre_holdback_scale", 1.0)),
                "leader_post_scale": float(getattr(hint, "leader_post_scale", 0.25)),
                "follower_post_scale": float(getattr(hint, "follower_post_scale", 0.5)),
            }
        )
    return hints


def _smooth_window_weight(payload_y_m: float, y_start_m: float, y_end_m: float) -> float:
    if y_end_m <= y_start_m:
        return 0.0
    if payload_y_m <= y_start_m or payload_y_m >= y_end_m:
        return 0.0
    alpha = (payload_y_m - y_start_m) / max(y_end_m - y_start_m, 1e-9)
    return float(np.sin(np.pi * alpha))


def apply_passage_hints_to_reference(
    reference_uav_positions_m: np.ndarray,
    payload_positions_m: np.ndarray,
    passage_hints: list[dict[str, float | int | str]],
) -> np.ndarray:
    updated_positions_m = np.asarray(reference_uav_positions_m, dtype=float).copy()
    payload_positions_m = np.asarray(payload_positions_m, dtype=float)
    if not passage_hints:
        return updated_positions_m
    for sample_index in range(updated_positions_m.shape[0]):
        payload_y_m = float(payload_positions_m[sample_index, 1])
        payload_z_m = float(payload_positions_m[sample_index, 2])
        for hint in passage_hints:
            y_start_m = float(hint["y_start"])
            y_end_m = float(hint["y_end"])
            y_center_m = float(hint["y_center"])
            weight = _smooth_window_weight(payload_y_m, y_start_m, y_end_m)
            if weight <= 0.0:
                continue
            leader_index = int(hint["leader_index"])
            forward_offset_m = float(hint["forward_offset_m"])
            holdback_offset_m = float(hint["holdback_offset_m"])
            flatten_scale = float(hint["flatten_scale"])
            leader_pre_scale = float(hint["leader_pre_scale"])
            follower_pre_holdback_scale = float(hint["follower_pre_holdback_scale"])
            leader_post_scale = float(hint["leader_post_scale"])
            follower_post_scale = float(hint["follower_post_scale"])
            kind = str(hint["kind"])
            pre_center = payload_y_m < y_center_m

            if kind == "horizontal_pass":
                effective_flatten_scale = max(0.18, 0.8 * flatten_scale)
                updated_positions_m[sample_index, :, 2] = payload_z_m + effective_flatten_scale * (updated_positions_m[sample_index, :, 2] - payload_z_m)
                updated_positions_m[sample_index, :, 0] = payload_positions_m[sample_index, 0] + effective_flatten_scale * (updated_positions_m[sample_index, :, 0] - payload_positions_m[sample_index, 0])

            for uav_index in range(updated_positions_m.shape[1]):
                if kind == "vertical_pass":
                    if pre_center:
                        if uav_index == leader_index:
                            updated_positions_m[sample_index, uav_index, 1] += leader_pre_scale * weight * forward_offset_m
                        else:
                            updated_positions_m[sample_index, uav_index, 1] -= follower_pre_holdback_scale * weight * holdback_offset_m
                    else:
                        if uav_index == leader_index:
                            updated_positions_m[sample_index, uav_index, 1] += leader_post_scale * weight * forward_offset_m
                        else:
                            updated_positions_m[sample_index, uav_index, 1] += follower_post_scale * weight * forward_offset_m
                elif kind == "horizontal_pass":
                    if pre_center:
                        if uav_index == leader_index:
                            updated_positions_m[sample_index, uav_index, 1] += leader_pre_scale * weight * forward_offset_m
                        else:
                            updated_positions_m[sample_index, uav_index, 1] -= follower_pre_holdback_scale * weight * holdback_offset_m
                    else:
                        if uav_index == leader_index:
                            updated_positions_m[sample_index, uav_index, 1] += leader_post_scale * weight * forward_offset_m
                        else:
                            updated_positions_m[sample_index, uav_index, 1] += follower_post_scale * weight * forward_offset_m
                else:
                    if uav_index == leader_index:
                        updated_positions_m[sample_index, uav_index, 1] += weight * forward_offset_m
                    else:
                        updated_positions_m[sample_index, uav_index, 1] -= weight * holdback_offset_m
    return updated_positions_m


def load_planner_trajectory(mat_path: Path | None, sample_t_path: Path | None, sample_xyz_path: Path | None) -> TrajectorySamples3D:
    if mat_path is not None:
        mat = scipy.io.loadmat(mat_path)
        traj = mat.get("traj", None)
        if traj is None:
            raise ValueError("corridor_export.mat does not contain a 'traj' field")
        sample_t = np.asarray(traj[0][0]["sample_t"], dtype=float).reshape(-1)
        sample_xyz = np.asarray(traj[0][0]["sample_xyz"], dtype=float)
        roll_t = _extract_roll_from_mat(mat, sample_t)
        sample_t, sample_xyz, roll_t = _deduplicate_time_aligned_series(sample_t, sample_xyz, roll_t)
        orientation_rpy_rad = np.column_stack([roll_t, np.zeros_like(roll_t), np.zeros_like(roll_t)])
        return TrajectorySamples3D.from_position_samples(
            time_s=sample_t,
            position_m=sample_xyz,
            orientation_rpy_rad=orientation_rpy_rad,
            source=f"planner_mat::{mat_path.name}",
        )

    if sample_t_path is None or sample_xyz_path is None:
        raise ValueError("either mat_path or both sample_t_path/sample_xyz_path must be provided")
    sample_t = np.load(sample_t_path)
    sample_xyz = np.load(sample_xyz_path)
    sample_t, sample_xyz = _deduplicate_time_aligned_series(sample_t, sample_xyz)
    return TrajectorySamples3D.from_position_samples(
        time_s=sample_t,
        position_m=sample_xyz,
        source=f"planner_npy::{sample_t_path.name}+{sample_xyz_path.name}",
    )


def build_rotation_series(trajectory: TrajectorySamples3D) -> np.ndarray:
    orientation_rpy_rad = np.asarray(trajectory.orientation_rpy_rad, dtype=float)
    return np.asarray([rotation_matrix_from_rpy(rpy) for rpy in orientation_rpy_rad], dtype=float)


def save_plots(output_dir: Path, trajectory: TrajectorySamples3D, series, exported_uav_positions_m: np.ndarray) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_idx, axis in enumerate(axes):
        axis.plot(trajectory.time_s, trajectory.position_m[:, axis_idx], linewidth=2.0)
        axis.set_ylabel(f"payload {labels[axis_idx]} [m]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path = output_dir / "planner_payload_trajectory_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for comp_idx, name in enumerate(labels):
        axes[0].plot(series.time_s, series.payload_wrench_body[:, comp_idx], label=f"F_{name}")
        axes[1].plot(series.time_s, series.payload_wrench_body[:, 3 + comp_idx], label=f"M_{name}")
    axes[0].set_ylabel("force body [N]")
    axes[1].set_ylabel("torque body [Nm]")
    axes[1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "planner_payload_wrench_body.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for uav_idx in range(series.attachment_force_body_n.shape[1]):
        axes[0].plot(series.time_s, series.attachment_force_body_n[:, uav_idx, 0], label=f"uav{uav_idx+1}")
        axes[1].plot(series.time_s, series.attachment_force_body_n[:, uav_idx, 1], label=f"uav{uav_idx+1}")
        axes[2].plot(series.time_s, series.attachment_force_body_n[:, uav_idx, 2], label=f"uav{uav_idx+1}")
    axes[0].set_ylabel("f_x [N]")
    axes[1].set_ylabel("f_y [N]")
    axes[2].set_ylabel("f_z [N]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "planner_attachment_force_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    reference_uav_positions_m = np.asarray(series.quadrotor_position_inertial_m, dtype=float)
    exported_uav_positions_m = np.asarray(exported_uav_positions_m, dtype=float)
    for uav_idx in range(reference_uav_positions_m.shape[1]):
        axes[0].plot(series.time_s, reference_uav_positions_m[:, uav_idx, 0], linestyle="--", alpha=0.6, label=f"ref_uav{uav_idx+1}")
        axes[1].plot(series.time_s, reference_uav_positions_m[:, uav_idx, 1], linestyle="--", alpha=0.6, label=f"ref_uav{uav_idx+1}")
        axes[2].plot(series.time_s, reference_uav_positions_m[:, uav_idx, 2], linestyle="--", alpha=0.6, label=f"ref_uav{uav_idx+1}")
        axes[0].plot(series.time_s, exported_uav_positions_m[:, uav_idx, 0], linewidth=2.0, label=f"play_uav{uav_idx+1}")
        axes[1].plot(series.time_s, exported_uav_positions_m[:, uav_idx, 1], linewidth=2.0, label=f"play_uav{uav_idx+1}")
        axes[2].plot(series.time_s, exported_uav_positions_m[:, uav_idx, 2], linewidth=2.0, label=f"play_uav{uav_idx+1}")
    axes[0].set_ylabel("uav x [m]")
    axes[1].set_ylabel("uav y [m]")
    axes[2].set_ylabel("uav z [m]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "planner_uav_position_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def save_run_summary(output_dir: Path, summary: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "planner_run_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    total_start_s = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run cable-force inference on real planner export files")
    parser.add_argument("--mat", type=Path, default=Path("corridor_export.mat"), help="Path to corridor_export.mat")
    parser.add_argument("--sample-t", type=Path, default=None, help="Optional sample_t npy path")
    parser.add_argument("--sample-xyz", type=Path, default=None, help="Optional sample_xyz npy path")
    parser.add_argument("--mass-kg", type=float, default=1.6, help="Payload mass")
    parser.add_argument("--box-size", type=float, nargs=3, default=[0.8, 0.3, 0.2], help="Box size xyz in meters")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inverse_transport_development/results/planner_export_inference"),
        help="Directory used to save plots and npz outputs",
    )
    parser.add_argument("--baseline-only", action="store_true", help="Export the minimum-norm baseline UAV trajectory without sequential refinement")
    parser.add_argument("--corridor-half-width", type=float, nargs=3, default=[1.50, 1.50, 1.50], help="Local corridor half width xyz for sequential UAV refinement")
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum sequential planner iterations")
    parser.add_argument("--scene-refinement-passes", type=int, default=4, help="Number of repeated fixed-scene planner passes, each using the previous pass output as the next reference")
    parser.add_argument("--trust-region-half-width", type=float, default=1.50, help="Trust-region half width used by the local UAV QP")
    parser.add_argument("--max-speed-mps", type=float, default=8.0, help="Maximum discrete UAV speed limit used by the local QP")
    parser.add_argument("--max-acceleration-mps2", type=float, default=30.0, help="Maximum discrete UAV acceleration limit used by the local QP")
    parser.add_argument("--door-ring-margin", type=float, default=0.55, help="Door wall ring margin used to reconstruct passable obstacles from corridor_export.mat")
    parser.add_argument("--disable-fixed-scene-obstacles", action="store_true", help="Disable loading doors/forbidden from corridor_export.mat into the sequential planner")
    args = parser.parse_args()

    mat_path = args.mat if args.mat is not None and args.mat.exists() else None
    sample_t_path = args.sample_t if args.sample_t is not None and args.sample_t.exists() else None
    sample_xyz_path = args.sample_xyz if args.sample_xyz is not None and args.sample_xyz.exists() else None

    if mat_path is None and (sample_t_path is None or sample_xyz_path is None):
        raise FileNotFoundError(
            "No real planner export artifacts were found. Expected corridor_export.mat or both sample_t/sample_xyz npy files."
        )

    load_start_s = time.perf_counter()
    trajectory = load_planner_trajectory(mat_path=mat_path, sample_t_path=sample_t_path, sample_xyz_path=sample_xyz_path)
    trajectory_load_time_s = time.perf_counter() - load_start_s
    params = PayloadPhysicalParams(
        mass_kg=args.mass_kg,
        geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.asarray(args.box_size, dtype=float)),
    )
    wrench_start_s = time.perf_counter()
    wrench = compute_payload_wrench(trajectory, params)
    attachments = build_three_uav_box_attachment_set(np.asarray(args.box_size, dtype=float))
    series = infer_cable_force_series(trajectory, wrench, attachments)
    wrench_inference_time_s = time.perf_counter() - wrench_start_s
    exported_uav_positions_m = np.asarray(series.quadrotor_position_inertial_m, dtype=float)
    planner_result = None
    fixed_scene_obstacles: list[InflatedObstaclePrismXZ] = []
    passage_hints: list[dict[str, float | int | str]] = []
    scene_setup_start_s = time.perf_counter()
    if mat_path is not None and not args.disable_fixed_scene_obstacles:
        fixed_scene_obstacles = build_fixed_scene_obstacles(mat_path, door_ring_margin_m=float(args.door_ring_margin))
        passage_hints = load_passage_hints(mat_path)
    fixed_scene_setup_time_s = time.perf_counter() - scene_setup_start_s
    reference_shaping_time_s = 0.0
    sequential_refinement_time_s = 0.0
    if not args.baseline_only:
        reference_shaping_start_s = time.perf_counter()
        current_reference_uav_positions_m = apply_passage_hints_to_reference(
            reference_uav_positions_m=exported_uav_positions_m,
            payload_positions_m=trajectory.position_m,
            passage_hints=passage_hints,
        )
        reference_shaping_time_s = time.perf_counter() - reference_shaping_start_s
        n_scene_passes = max(int(args.scene_refinement_passes), 1)
        best_scene_pass_positions_m = current_reference_uav_positions_m.copy()
        best_scene_pass_result = None
        best_scene_pass_score = None
        sequential_refinement_start_s = time.perf_counter()
        for scene_pass_index in range(n_scene_passes):
            planner_result = plan_collision_aware_uav_positions(
                time_s=trajectory.time_s,
                payload_positions_m=trajectory.position_m,
                payload_rotation_matrices=build_rotation_series(trajectory),
                reference_uav_positions_m=current_reference_uav_positions_m,
                wrench_body_series=np.asarray(series.payload_wrench_body, dtype=float),
                attachments=attachments,
                config=SequentialPlanningConfig(
                    corridor_config=CorridorGenerationConfig(half_width_xyz_m=np.asarray(args.corridor_half_width, dtype=float)),
                    kinematic_limits=UAVKinematicLimits(max_speed_mps=args.max_speed_mps, max_acceleration_mps2=args.max_acceleration_mps2),
                    tension_max_n=np.array([45.0, 45.0, 45.0], dtype=float),
                    obstacles=tuple(fixed_scene_obstacles),
                    max_iterations=args.max_iterations,
                    length_tolerance_m=0.07,
                    force_residual_absolute_tolerance_n=0.8,
                    force_residual_relative_tolerance=0.08,
                    torque_residual_absolute_tolerance_nm=0.18,
                    torque_residual_relative_tolerance=3.00,
                    force_relative_scale_floor_n=1.0,
                    torque_relative_scale_floor_nm=0.05,
                    local_reference_weight=1.0,
                    tension_surrogate_weight=0.25,
                    tension_linearization_weight=0.50,
                    local_shape_weight=0.05,
                    local_smoothness_weight=0.05,
                    min_separation_m=0.10,
                    trust_region_half_width_m=args.trust_region_half_width,
                    tension_linearization_delta_m=1e-3,
                    feedback_force_gain_m_per_n=0.0,
                    feedback_torque_gain_m_per_nm=0.10,
                    feedback_max_step_m=0.04,
                ),
            )
            scene_pass_score = (
                -int(planner_result.cable_collision_count),
                -int(planner_result.obstacle_collision_count),
                float(planner_result.min_cable_clearance_m),
                float(planner_result.min_obstacle_clearance_m),
                -float(planner_result.mean_residual_norm),
            )
            if best_scene_pass_result is None or best_scene_pass_score is None or scene_pass_score > best_scene_pass_score:
                best_scene_pass_result = planner_result
                best_scene_pass_score = scene_pass_score
                best_scene_pass_positions_m = np.asarray(planner_result.positions_m, dtype=float)
            current_reference_uav_positions_m = np.asarray(planner_result.positions_m, dtype=float)
            print(
                "[planner_inference] "
                f"fixed_scene_pass={scene_pass_index + 1}/{n_scene_passes} "
                f"status={planner_result.status} iterations={planner_result.iterations_run} "
                f"mean_residual_norm={planner_result.mean_residual_norm:.3e} "
                f"obstacle_collisions={planner_result.obstacle_collision_count} "
                f"min_clearance={planner_result.min_obstacle_clearance_m:.3e} "
                f"cable_collisions={planner_result.cable_collision_count} "
                f"min_cable_clearance={planner_result.min_cable_clearance_m:.3e}"
            )
            if scene_pass_index >= 1 and best_scene_pass_score is not None and scene_pass_score < best_scene_pass_score:
                break
        if best_scene_pass_result is not None:
            planner_result = best_scene_pass_result
            exported_uav_positions_m = best_scene_pass_positions_m
        sequential_refinement_time_s = time.perf_counter() - sequential_refinement_start_s

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "planner_inference_series.npz",
        time_s=series.time_s,
        payload_position_m=trajectory.position_m,
        payload_orientation_rpy_rad=trajectory.orientation_rpy_rad,
        payload_theta_rad=trajectory.orientation_rpy_rad[:, 0],
        payload_wrench_body=series.payload_wrench_body,
        attachment_force_body_n=series.attachment_force_body_n,
        q_body=series.q_body,
        force_norm_n=series.force_norm_n,
        quadrotor_position_reference_inertial_m=series.quadrotor_position_inertial_m,
        quadrotor_position_inertial_m=exported_uav_positions_m,
        residual_wrench_body=series.residual_wrench_body,
        sequential_planner_mean_residual_norm=(np.nan if planner_result is None else planner_result.mean_residual_norm),
        sequential_planner_iterations_run=(0 if planner_result is None else planner_result.iterations_run),
        sequential_planner_residual_history=(np.zeros((0,), dtype=float) if planner_result is None else planner_result.residual_history),
    )
    plot_start_s = time.perf_counter()
    saved_paths = save_plots(args.output_dir, trajectory, series, exported_uav_positions_m=exported_uav_positions_m)
    plot_export_time_s = time.perf_counter() - plot_start_s

    residual_norm = np.linalg.norm(series.residual_wrench_body, axis=1)
    total_runtime_s = time.perf_counter() - total_start_s
    summary = {
        "source": trajectory.source,
        "num_samples": int(trajectory.time_s.size),
        "mode": ("minimum_norm_baseline" if planner_result is None else "sequential_planner"),
        "baseline_only": bool(args.baseline_only),
        "fixed_scene_obstacles": int(len(fixed_scene_obstacles)),
        "passage_hints": int(len(passage_hints)),
        "residual_norm": {
            "min": float(residual_norm.min()),
            "mean": float(residual_norm.mean()),
            "max": float(residual_norm.max()),
        },
        "timing_s": {
            "trajectory_load": float(trajectory_load_time_s),
            "wrench_inference": float(wrench_inference_time_s),
            "fixed_scene_setup": float(fixed_scene_setup_time_s),
            "reference_shaping": float(reference_shaping_time_s),
            "sequential_refinement": float(sequential_refinement_time_s),
            "plot_export": float(plot_export_time_s),
            "total": float(total_runtime_s),
        },
    }
    if planner_result is not None:
        summary["planner"] = {
            "status": str(planner_result.status),
            "iterations": int(planner_result.iterations_run),
            "mean_residual_norm": float(planner_result.mean_residual_norm),
            "obstacle_collisions": int(planner_result.obstacle_collision_count),
            "min_obstacle_clearance_m": float(planner_result.min_obstacle_clearance_m),
            "cable_collisions": int(planner_result.cable_collision_count),
            "min_cable_clearance_m": float(planner_result.min_cable_clearance_m),
        }
    summary_path = save_run_summary(args.output_dir, summary)
    print(f"[planner_inference] source={trajectory.source} samples={trajectory.time_s.size}")
    print(
        "[planner_inference] residual_norm[min/mean/max]="
        f"{residual_norm.min():.3e}/{residual_norm.mean():.3e}/{residual_norm.max():.3e}"
    )
    if planner_result is None:
        print("[planner_inference] exported_uav_source=minimum_norm_baseline")
    else:
        print(f"[planner_inference] fixed_scene_obstacles={len(fixed_scene_obstacles)}")
        print(f"[planner_inference] passage_hints={len(passage_hints)}")
        print(
            "[planner_inference] "
            f"exported_uav_source=sequential_planner status={planner_result.status} "
            f"iterations={planner_result.iterations_run} "
            f"mean_residual_norm={planner_result.mean_residual_norm:.3e} "
            f"obstacle_collisions={planner_result.obstacle_collision_count} "
            f"min_clearance={planner_result.min_obstacle_clearance_m:.3e} "
            f"cable_collisions={planner_result.cable_collision_count} "
            f"min_cable_clearance={planner_result.min_cable_clearance_m:.3e}"
        )
    for path in saved_paths:
        print(f"[planner_inference] saved {path}")
    print(f"[planner_inference] saved {args.output_dir / 'planner_inference_series.npz'}")
    print(f"[planner_inference] saved {summary_path}")


if __name__ == "__main__":
    main()