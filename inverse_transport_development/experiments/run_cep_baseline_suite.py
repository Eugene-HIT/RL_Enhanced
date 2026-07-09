# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-07-09
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-07-09
# 目的 / Purpose:
#   运行 CEP 第一版 baseline 对比实验，统一调度主方法、minimum-norm baseline
#   和无固定场景障碍对照，并导出结构化汇总结果。
#   Run the first CEP baseline comparison suite, covering the main method,
#   the minimum-norm baseline, and a no-fixed-scene-obstacle control run,
#   then export a structured summary.
# 主要输入 / Main Inputs:
#   fixed-scene MAT 文件、主实验输出目录、door ring margin、payload box size。
#   A fixed-scene MAT file, output directory, door ring margin, and payload
#   box size.
# 主要输出 / Main Outputs:
#   每个 baseline 的实验子目录，以及汇总 JSON/CSV 表。
#   Per-baseline result folders plus summary JSON/CSV tables.
# ==============================================

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
	sys.path.insert(0, str(PKG_ROOT))

from src.common import build_three_uav_box_attachment_set, rotation_matrix_from_rpy
from src.uav_pose_inference import InflatedObstaclePrismXZ, evaluate_cable_obstacle_collisions

from experiments.plot_fixed_scene_uav_paths import load_fixed_scene_obstacles, obstacle_collision_count


def _compute_scene_metrics(mat_path: Path, npz_path: Path, door_ring_margin_m: float, box_size_xyz_m: np.ndarray) -> dict[str, float | int]:
	_, door_obstacles, forbidden_obstacles = load_fixed_scene_obstacles(mat_path, door_ring_margin_m=door_ring_margin_m)
	bundle = np.load(npz_path)
	payload_positions_m = np.asarray(bundle["payload_position_m"], dtype=float)
	payload_orientation_rpy_rad = np.asarray(bundle["payload_orientation_rpy_rad"], dtype=float)
	refined_positions_m = np.asarray(bundle["quadrotor_position_inertial_m"], dtype=float)
	attachments = build_three_uav_box_attachment_set(np.asarray(box_size_xyz_m, dtype=float))
	payload_rotation_matrices = np.asarray([rotation_matrix_from_rpy(rpy) for rpy in payload_orientation_rpy_rad], dtype=float)
	anchor_positions_m = np.zeros((payload_positions_m.shape[0], attachments.count, 3), dtype=float)
	for sample_index in range(payload_positions_m.shape[0]):
		anchor_positions_m[sample_index] = payload_positions_m[sample_index][None, :] + (payload_rotation_matrices[sample_index] @ attachments.r_i_body_m.T).T

	forbidden_collisions, forbidden_clearance_m = obstacle_collision_count(refined_positions_m, forbidden_obstacles)
	door_collisions, door_clearance_m = obstacle_collision_count(refined_positions_m, door_obstacles)
	all_scene_obstacles = [
		InflatedObstaclePrismXZ(
			polygon_xz=np.asarray(obstacle.polygon_xz.exterior.coords[:-1], dtype=float),
			y_min_m=float(obstacle.y_min),
			y_max_m=float(obstacle.y_max),
			label=obstacle.label,
		)
		for obstacle in [*door_obstacles, *forbidden_obstacles]
	]
	cable_collisions, cable_clearance_m = evaluate_cable_obstacle_collisions(anchor_positions_m, refined_positions_m, all_scene_obstacles)
	return {
		"forbidden_collisions": int(forbidden_collisions),
		"forbidden_min_clearance_m": float(forbidden_clearance_m),
		"door_collisions": int(door_collisions),
		"door_min_clearance_m": float(door_clearance_m),
		"cable_collisions": int(cable_collisions),
		"cable_min_clearance_m": float(cable_clearance_m),
	}


def _run_single_case(case_name: str, mat_path: Path, output_dir: Path, door_ring_margin_m: float, box_size_xyz_m: np.ndarray, extra_args: list[str]) -> dict:
	planner_script = THIS_DIR / "run_planner_export_inference.py"
	command = [
		sys.executable,
		str(planner_script),
		"--mat",
		str(mat_path),
		"--output-dir",
		str(output_dir),
		"--door-ring-margin",
		str(door_ring_margin_m),
	]
	command.extend(extra_args)
	subprocess.run(command, check=True)
	summary_path = output_dir / "planner_run_summary.json"
	if not summary_path.exists():
		raise FileNotFoundError(f"missing planner summary: {summary_path}")
	summary = json.loads(summary_path.read_text(encoding="utf-8"))
	npz_path = output_dir / "planner_inference_series.npz"
	scene_metrics = _compute_scene_metrics(
		mat_path=mat_path,
		npz_path=npz_path,
		door_ring_margin_m=door_ring_margin_m,
		box_size_xyz_m=box_size_xyz_m,
	)
	merged = {
		"case": case_name,
		"output_dir": str(output_dir),
		**summary,
		"scene_metrics": scene_metrics,
	}
	return merged


def _flatten_case_summary(case_summary: dict) -> dict[str, str | float | int]:
	planner = case_summary.get("planner", {})
	timing = case_summary.get("timing_s", {})
	scene_metrics = case_summary.get("scene_metrics", {})
	residual = case_summary.get("residual_norm", {})
	return {
		"case": str(case_summary["case"]),
		"mode": str(case_summary.get("mode", "")),
		"planner_status": str(planner.get("status", "baseline_only")),
		"iterations": int(planner.get("iterations", 0)),
		"residual_mean": float(residual.get("mean", float("nan"))),
		"door_collisions": int(scene_metrics.get("door_collisions", 0)),
		"door_min_clearance_m": float(scene_metrics.get("door_min_clearance_m", float("nan"))),
		"forbidden_collisions": int(scene_metrics.get("forbidden_collisions", 0)),
		"forbidden_min_clearance_m": float(scene_metrics.get("forbidden_min_clearance_m", float("nan"))),
		"cable_collisions": int(scene_metrics.get("cable_collisions", 0)),
		"cable_min_clearance_m": float(scene_metrics.get("cable_min_clearance_m", float("nan"))),
		"timing_total_s": float(timing.get("total", float("nan"))),
		"timing_refinement_s": float(timing.get("sequential_refinement", float("nan"))),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Run the first CEP baseline suite for the inverse planner")
	parser.add_argument(
		"--mat",
		type=Path,
		default=Path("inverse_transport_development/results/simple_passage_scene/simple_passage_scene.mat"),
		help="Path to the fixed-scene MAT file",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("inverse_transport_development/results/cep_revision/baselines"),
		help="Directory used to store baseline outputs and summaries",
	)
	parser.add_argument("--door-ring-margin", type=float, default=0.0, help="Door wall ring margin used by all baseline runs")
	parser.add_argument("--box-size", type=float, nargs=3, default=[0.8, 0.3, 0.2], help="Payload box size used for cable collision evaluation")
	args = parser.parse_args()

	args.output_root.mkdir(parents=True, exist_ok=True)
	box_size_xyz_m = np.asarray(args.box_size, dtype=float)
	baseline_cases = [
		("minimum_norm_baseline", ["--baseline-only"]),
		("sequential_no_scene_obstacles", ["--disable-fixed-scene-obstacles"]),
		("sequential_fixed_scene", []),
	]
	case_summaries: list[dict] = []
	for case_name, extra_args in baseline_cases:
		case_output_dir = args.output_root / case_name
		case_summary = _run_single_case(
			case_name=case_name,
			mat_path=args.mat,
			output_dir=case_output_dir,
			door_ring_margin_m=float(args.door_ring_margin),
			box_size_xyz_m=box_size_xyz_m,
			extra_args=extra_args,
		)
		case_summaries.append(case_summary)
		flat = _flatten_case_summary(case_summary)
		print(
			"[cep_baseline_suite] "
			f"case={flat['case']} status={flat['planner_status']} "
			f"door_collisions={flat['door_collisions']} cable_collisions={flat['cable_collisions']} "
			f"timing_total_s={flat['timing_total_s']:.3f}"
		)

	json_path = args.output_root / "baseline_suite_summary.json"
	json_path.write_text(json.dumps(case_summaries, indent=2, sort_keys=True), encoding="utf-8")
	flat_rows = [_flatten_case_summary(item) for item in case_summaries]
	csv_path = args.output_root / "baseline_suite_summary.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
		writer.writeheader()
		writer.writerows(flat_rows)
	print(f"[cep_baseline_suite] saved {json_path}")
	print(f"[cep_baseline_suite] saved {csv_path}")


if __name__ == "__main__":
	main()