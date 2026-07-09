# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-25
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-25
# 目的 / Purpose:
#   对第一版轨迹与 wrench 框架做最小可执行验证。
#   Provide minimal executable checks for the first-stage trajectory and wrench
#   framework.
# 主要输入 / Main Inputs:
#   内置解析轨迹与物理参数。
#   Built-in analytic trajectories and physical parameters.
# 主要输出 / Main Outputs:
#   unittest 断言结果。
#   unittest assertion results.
# ==============================================

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.cable_force_inference import (
    CableForceInferenceSeries,
    PayloadGeometry,
    PayloadPhysicalParams,
    allocate_attachment_forces,
    compute_payload_wrench,
    compute_translational_wrench,
    forces_to_cable_state,
    infer_cable_force_series,
    inertia_matrix_for_box,
)
from src.common import AttachmentKinematicsInput, RigidBodyLoadAttachmentSet, build_three_uav_box_attachment_set
from src.common.trajectory import PiecewisePolynomialTrajectory3D, TrajectorySamples3D


def build_demo_trajectory_for_inference() -> TrajectorySamples3D:
    time_s = np.linspace(0.0, 2.0, 81)
    position_m = np.column_stack(
        [
            0.4 * time_s * time_s,
            0.15 * np.sin(2.0 * np.pi * time_s / 2.0),
            0.6 + 0.05 * np.cos(2.0 * np.pi * time_s / 2.0),
        ]
    )
    orientation_rpy_rad = np.column_stack(
        [
            0.08 * time_s * time_s,
            0.03 * np.sin(2.0 * np.pi * time_s / 2.0),
            0.02 * np.cos(2.0 * np.pi * time_s / 2.0),
        ]
    )
    return TrajectorySamples3D.from_position_samples(
        time_s=time_s,
        position_m=position_m,
        orientation_rpy_rad=orientation_rpy_rad,
        source="test_demo_inference",
    )


def generate_demo_cable_force_series() -> tuple[TrajectorySamples3D, CableForceInferenceSeries]:
    trajectory = build_demo_trajectory_for_inference()
    params = PayloadPhysicalParams(
        mass_kg=1.6,
        geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.array([0.8, 0.3, 0.2], dtype=float)),
    )
    wrench = compute_payload_wrench(trajectory, params)
    attachments = build_three_uav_box_attachment_set(np.array([0.8, 0.3, 0.2], dtype=float))
    series = infer_cable_force_series(trajectory, wrench, attachments)
    return trajectory, series


def save_demo_plots(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory, series = generate_demo_cable_force_series()
    saved_paths: list[Path] = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x", "y", "z"]
    for axis_idx, axis in enumerate(axes):
        axis.plot(trajectory.time_s, trajectory.position_m[:, axis_idx], linewidth=2.0)
        axis.set_ylabel(f"payload {labels[axis_idx]} [m]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path = output_dir / "payload_trajectory_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis_idx, axis in enumerate(axes[0:1]):
        pass
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
    path = output_dir / "payload_wrench_body.png"
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
    path = output_dir / "attachment_force_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for uav_idx in range(series.quadrotor_position_inertial_m.shape[1]):
        axes[0].plot(series.time_s, series.quadrotor_position_inertial_m[:, uav_idx, 0], label=f"uav{uav_idx+1}")
        axes[1].plot(series.time_s, series.quadrotor_position_inertial_m[:, uav_idx, 1], label=f"uav{uav_idx+1}")
        axes[2].plot(series.time_s, series.quadrotor_position_inertial_m[:, uav_idx, 2], label=f"uav{uav_idx+1}")
    axes[0].set_ylabel("uav x [m]")
    axes[1].set_ylabel("uav y [m]")
    axes[2].set_ylabel("uav z [m]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "uav_position_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


class TrajectoryAndWrenchTests(unittest.TestCase):
    def test_time_series_inference_recovers_uav_positions_and_small_residual(self) -> None:
        trajectory, series = generate_demo_cable_force_series()
        self.assertEqual(series.attachment_force_body_n.shape, (trajectory.time_s.size, 3, 3))
        self.assertEqual(series.q_body.shape, (trajectory.time_s.size, 3, 3))
        self.assertEqual(series.quadrotor_position_inertial_m.shape, (trajectory.time_s.size, 3, 3))
        np.testing.assert_allclose(np.linalg.norm(series.q_body, axis=2), 1.0, atol=1e-6)
        np.testing.assert_allclose(series.residual_wrench_body, 0.0, atol=1e-6)

    def test_three_uav_box_attachment_preset_matches_requested_layout(self) -> None:
        attachments = build_three_uav_box_attachment_set(np.array([2.0, 1.0, 4.0], dtype=float))
        expected = np.array(
            [
                [-1.0, 0.0, -2.0],
                [1.0, 0.0, -2.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(attachments.r_i_body_m, expected, atol=1e-12)

    def test_attachment_force_allocation_reconstructs_wrench(self) -> None:
        attachments = build_three_uav_box_attachment_set(np.array([2.0, 1.0, 4.0], dtype=float))
        wrench_body = np.array([2.0, -1.0, 6.0, 0.0, 8.0, 0.0], dtype=float)
        result = allocate_attachment_forces(wrench_body, attachments)
        np.testing.assert_allclose(result.reconstructed_wrench_body, wrench_body, atol=1e-8)
        np.testing.assert_allclose(result.residual_wrench_body, 0.0, atol=1e-8)

        q_body, tensions_n = forces_to_cable_state(result.attachment_force_body_n)
        reconstructed_force = -tensions_n[:, None] * q_body
        np.testing.assert_allclose(reconstructed_force, result.attachment_force_body_n, atol=1e-8)

    def test_attachment_kinematics_match_paper_equation_five(self) -> None:
        attachments = RigidBodyLoadAttachmentSet(
            r_i_body_m=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
            cable_lengths_m=np.array([2.0, 3.0], dtype=float),
        )
        kinematics = AttachmentKinematicsInput(
            x_L=np.array([10.0, 20.0, 30.0], dtype=float),
            R_L=np.eye(3, dtype=float),
            q_body=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float),
        )
        x_i = attachments.quadrotor_positions(kinematics)
        expected = np.array([[11.0, 20.0, 28.0], [10.0, 19.0, 30.0]], dtype=float)
        np.testing.assert_allclose(x_i, expected, atol=1e-12)

    def test_attachment_wrench_map_matches_body_frame_definition(self) -> None:
        attachments = RigidBodyLoadAttachmentSet(
            r_i_body_m=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
            cable_lengths_m=np.array([2.0, 3.0], dtype=float),
        )
        q_body = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
        phi = attachments.wrench_map_body(q_body)
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(phi, expected, atol=1e-12)

    def test_box_inertia_is_recovered_from_geometry(self) -> None:
        inertia = inertia_matrix_for_box(2.0, np.array([2.0, 4.0, 6.0], dtype=float))
        expected = np.diag([
            2.0 * (4.0 ** 2 + 6.0 ** 2) / 12.0,
            2.0 * (2.0 ** 2 + 6.0 ** 2) / 12.0,
            2.0 * (2.0 ** 2 + 4.0 ** 2) / 12.0,
        ])
        np.testing.assert_allclose(inertia, expected, atol=1e-12)

    def test_hover_from_position_samples(self) -> None:
        time_s = np.linspace(0.0, 1.0, 11)
        position_m = np.column_stack([np.zeros_like(time_s), np.zeros_like(time_s), np.ones_like(time_s)])
        trajectory = TrajectorySamples3D.from_position_samples(time_s=time_s, position_m=position_m)
        wrench = compute_translational_wrench(trajectory, PayloadPhysicalParams(mass_kg=2.0))
        np.testing.assert_allclose(wrench.force_n[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(wrench.force_n[:, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(wrench.force_n[:, 2], 19.62, atol=1e-6)

    def test_polynomial_sampling_produces_expected_acceleration(self) -> None:
        trajectory = PiecewisePolynomialTrajectory3D(
            coeffs_x=np.array([[0.0, 0.0, 0.5]], dtype=float),
            coeffs_y=np.array([[0.0, 0.0, 0.0]], dtype=float),
            coeffs_z=np.array([[0.0, 0.0, 0.0]], dtype=float),
            t_per_seg=np.array([1.0], dtype=float),
            order=2,
            source="unit_test_poly",
        ).sample(dt=0.1)
        np.testing.assert_allclose(trajectory.acceleration_mps2[:, 0], 1.0, atol=1e-9)
        np.testing.assert_allclose(trajectory.velocity_mps[0, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(trajectory.velocity_mps[-1, 0], 1.0, atol=1e-9)

    def test_translational_wrench_includes_gravity_and_acceleration(self) -> None:
        trajectory = PiecewisePolynomialTrajectory3D(
            coeffs_x=np.array([[0.0, 0.0, 1.0]], dtype=float),
            coeffs_y=np.array([[0.0, 0.0, 0.0]], dtype=float),
            coeffs_z=np.array([[0.0, 0.0, 0.0]], dtype=float),
            t_per_seg=np.array([1.0], dtype=float),
            order=2,
        ).sample(dt=0.2)
        wrench = compute_translational_wrench(trajectory, PayloadPhysicalParams(mass_kg=3.0))
        np.testing.assert_allclose(wrench.force_n[:, 0], 6.0, atol=1e-9)
        np.testing.assert_allclose(wrench.force_n[:, 2], 29.43, atol=1e-6)

    def test_payload_wrench_recovers_torque_from_angular_acceleration(self) -> None:
        time_s = np.linspace(0.0, 1.0, 11)
        position_m = np.zeros((time_s.size, 3), dtype=float)
        orientation_rpy_rad = np.column_stack([0.5 * time_s * time_s, np.zeros_like(time_s), np.zeros_like(time_s)])
        trajectory = TrajectorySamples3D.from_position_samples(
            time_s=time_s,
            position_m=position_m,
            velocity_mps=np.zeros((time_s.size, 3), dtype=float),
            acceleration_mps2=np.zeros((time_s.size, 3), dtype=float),
            orientation_rpy_rad=orientation_rpy_rad,
            source="rot_acc_test",
        )
        params = PayloadPhysicalParams(
            mass_kg=3.0,
            geometry=PayloadGeometry(shape_type="box", size_xyz_m=np.array([2.0, 4.0, 6.0], dtype=float)),
        )
        wrench = compute_payload_wrench(trajectory, params)
        expected_ixx = 3.0 * (4.0 ** 2 + 6.0 ** 2) / 12.0
        np.testing.assert_allclose(wrench.torque_nm[:, 0], expected_ixx, atol=1e-6)
        np.testing.assert_allclose(wrench.torque_nm[:, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(wrench.torque_nm[:, 2], 0.0, atol=1e-9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--plot-demo", action="store_true", help="Generate demo plots before running tests")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("inverse_transport_development/results/test_wrench_pipeline_plots"),
        help="Directory used to save demo plots",
    )
    args, remaining = parser.parse_known_args()

    if args.plot_demo:
        saved_paths = save_demo_plots(args.plot_dir)
        for path in saved_paths:
            print(f"[plot_demo] saved {path}")

    unittest.main(argv=[sys.argv[0], *remaining])
