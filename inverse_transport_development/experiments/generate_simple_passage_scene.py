# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-06-17
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-06-17
# 目的 / Purpose:
#   生成一个简化固定场景，包含 1 个实心障碍、1 个水平可通过区域和
#   1 个竖直可通过区域，并导出为与 corridor_export.mat 兼容的数据结构。
#   Generate a simplified fixed scene with one solid obstacle, one horizontal
#   passable opening, and one vertical passable opening, then export it in a
#   corridor_export.mat-compatible structure.
# 主要输入 / Main Inputs:
#   手工定义的场景几何、关键帧与姿态关键帧。
#   Manually defined scene geometry, payload keyframes, and roll waypoints.
# 主要输出 / Main Outputs:
#   simple_passage_scene.mat，包含 traj/keyframes/doors/forbidden/uav_passage_hints。
#   simple_passage_scene.mat containing traj/keyframes/doors/forbidden/uav_passage_hints.
# ==============================================

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


@dataclass(frozen=True)
class DoorRecord:
	name: str
	poly_xz: np.ndarray
	y_min: float
	y_max: float
	door_cx: float
	door_cz: float
	passage_kind: str


@dataclass(frozen=True)
class ForbiddenRecord:
	tag: str
	poly_xz: np.ndarray
	y_min: float
	y_max: float


@dataclass(frozen=True)
class PassageHintRecord:
	kind: str
	y_start: float
	y_end: float
	y_center: float
	leader_index: int
	forward_offset_m: float
	holdback_offset_m: float
	flatten_scale: float
	leader_pre_scale: float
	follower_pre_holdback_scale: float
	leader_post_scale: float
	follower_post_scale: float


def _structured_array(records: list[dict]) -> np.ndarray:
	if not records:
		return np.empty((0,), dtype=object)
	dtype = [(key, object) for key in records[0].keys()]
	array = np.empty((1, len(records)), dtype=dtype)
	for index, record in enumerate(records):
		for key, value in record.items():
			array[0, index][key] = value
	return array


def _make_rect(center_x: float, center_z: float, size_x: float, size_z: float) -> np.ndarray:
	half_x = 0.5 * size_x
	half_z = 0.5 * size_z
	return np.array(
		[
			[center_x - half_x, center_z - half_z],
			[center_x + half_x, center_z - half_z],
			[center_x + half_x, center_z + half_z],
			[center_x - half_x, center_z + half_z],
		],
		dtype=float,
	)


def _sample_polyline(points_xyz: np.ndarray, roll_wp: np.ndarray, dt: float = 0.05, nominal_speed_mps: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	points_xyz = np.asarray(points_xyz, dtype=float)
	roll_wp = np.asarray(roll_wp, dtype=float).reshape(-1)
	segment_lengths = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
	T_per_seg = np.maximum(segment_lengths / max(nominal_speed_mps, 1e-6), 0.8)
	sample_t = []
	sample_xyz = []
	t_global = 0.0
	for seg_index, T_seg in enumerate(T_per_seg):
		start = points_xyz[seg_index]
		end = points_xyz[seg_index + 1]
		n_step = max(2, int(np.ceil(T_seg / dt)))
		alphas = np.linspace(0.0, 1.0, n_step, dtype=float)
		for alpha in alphas:
			sample_t.append(t_global + alpha * T_seg)
			sample_xyz.append((1.0 - alpha) * start + alpha * end)
		t_global += T_seg
	return np.asarray(sample_t, dtype=float), np.asarray(sample_xyz, dtype=float), np.asarray(T_per_seg, dtype=float)


def build_scene_records() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	P_wp = np.array(
		[
			[0.0, -1.4, 0.92],
			[0.0, 1.3, 0.92],
			[0.0, 2.9, 0.92],
			[0.0, 4.6, 0.92],
			[0.0, 6.7, 0.92],
			[0.0, 8.9, 0.92],
		],
		dtype=float,
	)
	roll_wp = np.array([0.0, 0.0, 0.0, 0.10, 0.58, 0.0], dtype=float)
	hard = np.ones(P_wp.shape[0], dtype=np.uint8)
	tags = np.array(["start", "door_h_pre", "door_h_mid", "center_transition", "door_v_mid", "goal"], dtype=object)
	door_ids = np.array([-1, 0, 0, -1, 1, -1], dtype=np.int32)

	doors = [
		DoorRecord(
			name="door_horizontal",
			poly_xz=_make_rect(center_x=0.0, center_z=0.92, size_x=3.80, size_z=0.80),
			y_min=2.10,
			y_max=3.70,
			door_cx=0.0,
			door_cz=0.92,
			passage_kind="horizontal_pass",
		),
		DoorRecord(
			name="door_vertical",
			poly_xz=_make_rect(center_x=0.0, center_z=0.92, size_x=1.40, size_z=2.00),
			y_min=5.10,
			y_max=8.40,
			door_cx=0.0,
			door_cz=0.92,
			passage_kind="vertical_pass",
		),
	]
	forbidden: list[ForbiddenRecord] = []
	hints = [
		PassageHintRecord(
			kind="horizontal_pass",
			y_start=1.9,
			y_end=3.8,
			y_center=2.9,
			leader_index=0,
			forward_offset_m=0.55,
			holdback_offset_m=0.35,
			flatten_scale=0.35,
			leader_pre_scale=0.80,
			follower_pre_holdback_scale=0.75,
			leader_post_scale=0.15,
			follower_post_scale=0.55,
		),
		PassageHintRecord(
			kind="vertical_pass",
			y_start=5.25,
			y_end=8.05,
			y_center=6.7,
			leader_index=2,
			forward_offset_m=0.65,
			holdback_offset_m=0.40,
			flatten_scale=0.70,
			leader_pre_scale=1.10,
			follower_pre_holdback_scale=1.00,
			leader_post_scale=0.10,
			follower_post_scale=0.85,
		),
	]

	sample_t, sample_xyz, T_per_seg = _sample_polyline(P_wp, roll_wp)
	traj = _structured_array(
		[
			{
				"sample_t": sample_t,
				"sample_xyz": sample_xyz,
				"T_per_seg": T_per_seg,
			}
		]
	)
	keyframes = _structured_array(
		[
			{
				"P_wp": P_wp,
				"roll_wp": roll_wp,
				"tags": tags,
				"hard": hard,
				"door_ids": door_ids,
			}
		]
	)
	doors_array = _structured_array([
		{
			"name": door.name,
			"poly_xz": door.poly_xz,
			"y_min": float(door.y_min),
			"y_max": float(door.y_max),
			"door_cx": float(door.door_cx),
			"door_cz": float(door.door_cz),
			"passage_kind": door.passage_kind,
		}
		for door in doors
	])
	forbidden_array = _structured_array([
		{
			"tag": forbidden_item.tag,
			"poly_xz": forbidden_item.poly_xz,
			"y_min": float(forbidden_item.y_min),
			"y_max": float(forbidden_item.y_max),
		}
		for forbidden_item in forbidden
	])
	hints_array = _structured_array([
		{
			"kind": hint.kind,
			"y_start": float(hint.y_start),
			"y_end": float(hint.y_end),
			"y_center": float(hint.y_center),
			"leader_index": int(hint.leader_index),
			"forward_offset_m": float(hint.forward_offset_m),
			"holdback_offset_m": float(hint.holdback_offset_m),
			"flatten_scale": float(hint.flatten_scale),
			"leader_pre_scale": float(hint.leader_pre_scale),
			"follower_pre_holdback_scale": float(hint.follower_pre_holdback_scale),
			"leader_post_scale": float(hint.leader_post_scale),
			"follower_post_scale": float(hint.follower_post_scale),
		}
		for hint in hints
	])
	return traj, keyframes, doors_array, forbidden_array, hints_array


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate a simplified passage scene MAT file")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("inverse_transport_development/results/simple_passage_scene/simple_passage_scene.mat"),
		help="Output MAT file path",
	)
	args = parser.parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	traj, keyframes, doors_array, forbidden_array, hints_array = build_scene_records()
	scipy.io.savemat(
		args.output,
		{
			"traj": traj,
			"keyframes": keyframes,
			"doors": doors_array,
			"forbidden": forbidden_array,
			"uav_passage_hints": hints_array,
		},
		do_compression=True,
	)
	print(f"[simple_scene] saved {args.output}")


if __name__ == "__main__":
	main()