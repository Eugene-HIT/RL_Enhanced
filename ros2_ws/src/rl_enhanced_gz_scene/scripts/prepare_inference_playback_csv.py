#!/usr/bin/env python3
"""
创建时间: 2026-05-27
创建者: Eugene
最后修改时间: 2026-05-27
功能: 将逆推结果 npz 转换为 Gazebo smooth playback 主包使用的 14 列 CSV。
主要输入: planner_inference_series.npz，包含载荷位姿与三架无人机位置时序。
主要输出: playback CSV，列格式为 t,px,py,pz,theta,u1x,u1y,u1z,u2x,u2y,u2z,u3x,u3y,u3z。

Created: 2026-05-27
Author: Eugene
Last Modified: 2026-05-27
Purpose: Convert inverse-inference npz bundles into the 14-column CSV consumed by
the main Gazebo smooth playback package.
Main Inputs: planner_inference_series.npz with payload pose series and three UAV positions.
Main Outputs: playback CSV with columns
t,px,py,pz,theta,u1x,u1y,u1z,u2x,u2y,u2z,u3x,u3y,u3z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert inference npz bundle to playback CSV")
    parser.add_argument("--npz", required=True, help="Path to planner_inference_series.npz")
    parser.add_argument("--out", required=True, help="Output playback CSV path")
    args = parser.parse_args()

    bundle = np.load(args.npz)
    time_s = np.asarray(bundle["time_s"], dtype=float).reshape(-1)
    payload_position_m = np.asarray(bundle["payload_position_m"], dtype=float)
    payload_theta_rad = np.asarray(bundle["payload_theta_rad"], dtype=float).reshape(-1)
    quadrotor_position_inertial_m = np.asarray(bundle["quadrotor_position_inertial_m"], dtype=float)

    if payload_position_m.shape != (time_s.shape[0], 3):
        raise ValueError("payload_position_m must have shape (N, 3)")
    if payload_theta_rad.shape[0] != time_s.shape[0]:
        raise ValueError("payload_theta_rad must have shape (N,)")
    if quadrotor_position_inertial_m.shape != (time_s.shape[0], 3, 3):
        raise ValueError("quadrotor_position_inertial_m must have shape (N, 3, 3)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, time_value in enumerate(time_s):
            payload = payload_position_m[idx]
            drones = quadrotor_position_inertial_m[idx]
            row = [
                time_value,
                payload[0], payload[1], payload[2], payload_theta_rad[idx],
                drones[0, 0], drones[0, 1], drones[0, 2],
                drones[1, 0], drones[1, 1], drones[1, 2],
                drones[2, 0], drones[2, 1], drones[2, 2],
            ]
            handle.write(",".join(f"{value:.9f}" for value in row) + "\n")

    print(f"[prepare_inference_playback_csv] wrote {time_s.shape[0]} samples to {out_path}")


if __name__ == "__main__":
    main()