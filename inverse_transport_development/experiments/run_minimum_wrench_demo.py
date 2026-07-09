# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-25
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-25
# 目的 / Purpose:
#   演示第一版 wrench 框架如何从现有分段多项式轨迹或时间戳位置样本中恢复
#   载荷所需合力。
#   Demonstrate how the first-stage wrench framework recovers required payload
#   forces from either piecewise polynomial trajectories or sampled positions.
# 主要输入 / Main Inputs:
#   可选的 .npz 轨迹文件；若缺失，则使用内置的解析轨迹示例。
#   Optional .npz trajectory file; falls back to an analytic demo trajectory.
# 主要输出 / Main Outputs:
#   终端摘要与若干关键时刻的力值。
#   Console summary and representative force samples.
# ==============================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.cable_force_inference import PayloadGeometry, PayloadPhysicalParams, compute_payload_wrench
from src.common.trajectory import PiecewisePolynomialTrajectory3D, TrajectorySamples3D


def build_demo_polynomial() -> PiecewisePolynomialTrajectory3D:
    coeffs_x = np.array([[0.0, 0.0, 0.5], [0.5, 1.0, 0.0]], dtype=float)
    coeffs_y = np.zeros_like(coeffs_x)
    coeffs_z = np.array([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=float)
    return PiecewisePolynomialTrajectory3D(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        coeffs_z=coeffs_z,
        t_per_seg=np.array([1.0, 1.0], dtype=float),
        order=2,
        source="demo_piecewise_polynomial",
    )


def build_demo_orientation(time_s: np.ndarray) -> np.ndarray:
    roll = 0.2 * time_s * time_s
    pitch = np.zeros_like(time_s)
    yaw = np.zeros_like(time_s)
    return np.column_stack([roll, pitch, yaw])


def load_from_npz(npz_path: Path) -> TrajectorySamples3D:
    payload = np.load(npz_path)
    if {"coeffs_x", "coeffs_y", "coeffs_z", "T_per_seg", "order"}.issubset(payload.files):
        traj = PiecewisePolynomialTrajectory3D(
            coeffs_x=payload["coeffs_x"],
            coeffs_y=payload["coeffs_y"],
            coeffs_z=payload["coeffs_z"],
            t_per_seg=payload["T_per_seg"],
            order=int(payload["order"]),
            source=f"npz_polynomial::{npz_path.name}",
        )
        return traj.sample(dt=float(payload.get("dt", 0.08)))

    if {"sample_t", "sample_xyz"}.issubset(payload.files):
        return TrajectorySamples3D.from_position_samples(
            time_s=payload["sample_t"],
            position_m=payload["sample_xyz"],
            source=f"npz_samples::{npz_path.name}",
        )

    raise ValueError("npz file must contain either polynomial coeffs or sample_t/sample_xyz")


def summarize_wrench(force_n: np.ndarray, torque_nm: np.ndarray) -> str:
    force_norm = np.linalg.norm(force_n, axis=1)
    torque_norm = np.linalg.norm(torque_nm, axis=1)
    return (
        f"force_norm[min/mean/max]={force_norm.min():.3f}/"
        f"{force_norm.mean():.3f}/{force_norm.max():.3f} N, "
        f"torque_norm[min/mean/max]={torque_norm.min():.3f}/"
        f"{torque_norm.mean():.3f}/{torque_norm.max():.3f} Nm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimum payload wrench demo")
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional trajectory npz file")
    parser.add_argument("--mass-kg", type=float, default=1.0, help="Payload mass")
    parser.add_argument("--dt", type=float, default=0.08, help="Sampling dt for polynomial trajectories")
    parser.add_argument("--box-size", type=float, nargs=3, default=[0.8, 0.4, 0.2], help="Box size xyz in meters")
    parser.add_argument("--with-rotation-demo", action="store_true", help="Inject a simple orientation demo")
    args = parser.parse_args()

    if args.input_npz is not None:
        trajectory = load_from_npz(args.input_npz)
    else:
        trajectory = build_demo_polynomial().sample(dt=args.dt)

    if args.with_rotation_demo and trajectory.orientation_rpy_rad is None:
        trajectory = TrajectorySamples3D.from_position_samples(
            time_s=trajectory.time_s,
            position_m=trajectory.position_m,
            velocity_mps=trajectory.velocity_mps,
            acceleration_mps2=trajectory.acceleration_mps2,
            orientation_rpy_rad=build_demo_orientation(trajectory.time_s),
            source=f"{trajectory.source}::with_rotation_demo",
        )

    params = PayloadPhysicalParams(
        mass_kg=args.mass_kg,
        geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.asarray(args.box_size, dtype=float)),
    )
    wrench = compute_payload_wrench(trajectory, params)

    print(f"[wrench_demo] source={trajectory.source} samples={trajectory.time_s.size}")
    print(f"[wrench_demo] {summarize_wrench(wrench.force_n, wrench.torque_nm)}")
    probe_indices = np.linspace(0, trajectory.time_s.size - 1, num=min(5, trajectory.time_s.size), dtype=int)
    for idx in probe_indices:
        t_now = trajectory.time_s[idx]
        pos = trajectory.position_m[idx]
        acc = trajectory.acceleration_mps2[idx]
        force = wrench.force_n[idx]
        torque = wrench.torque_nm[idx]
        print(
            "[wrench_demo][sample] "
            f"t={t_now:.3f}s pos={np.array2string(pos, precision=3)} "
            f"acc={np.array2string(acc, precision=3)} "
            f"force={np.array2string(force, precision=3)} "
            f"torque={np.array2string(torque, precision=3)}"
        )


if __name__ == "__main__":
    main()
