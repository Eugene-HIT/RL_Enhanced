#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_required_array(bundle: np.lib.npyio.NpzFile, key: str, ndim: int) -> np.ndarray:
    if key not in bundle:
        raise KeyError(f"missing required field: {key}")
    array = np.asarray(bundle[key], dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"field {key} must have ndim={ndim}, got {array.ndim}")
    return array


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert inverse inference npz bundle to Gazebo playback CSV")
    parser.add_argument("--npz", required=True, help="Input planner_inference_series.npz path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Optional payload z offset baked into csv")
    args = parser.parse_args()

    bundle = np.load(args.npz)
    time_s = _load_required_array(bundle, "time_s", 1)
    payload_position_m = _load_required_array(bundle, "payload_position_m", 2)
    quadrotor_position_inertial_m = _load_required_array(bundle, "quadrotor_position_inertial_m", 3)

    if payload_position_m.shape != (time_s.size, 3):
        raise ValueError("payload_position_m must have shape (N, 3)")
    if quadrotor_position_inertial_m.shape != (time_s.size, 3, 3):
        raise ValueError("quadrotor_position_inertial_m must have shape (N, 3, 3)")

    if "payload_theta_rad" in bundle:
        payload_theta_rad = _load_required_array(bundle, "payload_theta_rad", 1)
    elif "payload_orientation_rpy_rad" in bundle:
        payload_orientation_rpy_rad = _load_required_array(bundle, "payload_orientation_rpy_rad", 2)
        if payload_orientation_rpy_rad.shape != (time_s.size, 3):
            raise ValueError("payload_orientation_rpy_rad must have shape (N, 3)")
        payload_theta_rad = payload_orientation_rpy_rad[:, 0]
    else:
        payload_theta_rad = np.zeros_like(time_s)

    if payload_theta_rad.shape != (time_s.size,):
        raise ValueError("payload_theta_rad must have shape (N,)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload_position_out = payload_position_m.copy()
    payload_position_out[:, 2] += float(args.z_offset)

    with out_path.open("w", encoding="ascii") as file_obj:
        file_obj.write(
            "# t,px,py,pz,theta,u1x,u1y,u1z,u2x,u2y,u2z,u3x,u3y,u3z\n"
        )
        for sample_idx in range(time_s.size):
            payload = payload_position_out[sample_idx]
            uavs = quadrotor_position_inertial_m[sample_idx]
            row = [
                float(time_s[sample_idx]),
                float(payload[0]),
                float(payload[1]),
                float(payload[2]),
                float(payload_theta_rad[sample_idx]),
                float(uavs[0, 0]),
                float(uavs[0, 1]),
                float(uavs[0, 2]),
                float(uavs[1, 0]),
                float(uavs[1, 1]),
                float(uavs[1, 2]),
                float(uavs[2, 0]),
                float(uavs[2, 1]),
                float(uavs[2, 2]),
            ]
            file_obj.write(",".join(f"{value:.9f}" for value in row) + "\n")

    print(f"[prepare_inference_playback_csv] wrote {time_s.size} samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())