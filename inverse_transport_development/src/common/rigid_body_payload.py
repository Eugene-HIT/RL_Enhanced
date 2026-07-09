 # -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-26
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-26
# 目的 / Purpose:
#   按经典多机吊载论文的符号体系定义刚体载荷挂点几何接口，显式表达
#   r_i、L_i、q_i，以及由此恢复无人机位置和载荷 wrench 映射矩阵。
#   Define rigid-body payload attachment geometry interfaces following the
#   classical cooperative aerial load-transport notation, exposing r_i, L_i,
#   q_i, and the induced quadrotor-position / wrench-mapping relations.
# 主要输入 / Main Inputs:
#   载荷体坐标系下挂点位置、绳长、绳索方向、载荷位置与姿态。
#   Attachment points in the payload body frame, cable lengths, cable
#   directions, and payload pose.
# 主要输出 / Main Outputs:
#   无人机位置、wrench 映射矩阵 Phi。
#   Quadrotor positions and the wrench mapping matrix Phi.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_array(name: str, value: np.ndarray, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(v) for v in shape_tail)})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix [v]_x such that [v]_x w = v x w."""

    vx, vy, vz = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]],
        dtype=float,
    )


def rotation_matrix_from_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    """Build a ZYX rotation matrix from roll-pitch-yaw angles."""

    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    r_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_y @ r_x


@dataclass(frozen=True)
class AttachmentKinematicsInput:
    """Payload pose and cable directions required by the rigid-body geometry model."""

    x_L: np.ndarray
    R_L: np.ndarray
    q_body: np.ndarray

    def __post_init__(self) -> None:
        x_L = np.asarray(self.x_L, dtype=float).reshape(-1)
        R_L = np.asarray(self.R_L, dtype=float)
        q_body = _validate_array("q_body", self.q_body, (3,))
        if x_L.shape != (3,) or not np.all(np.isfinite(x_L)):
            raise ValueError("x_L must be a finite 3-vector")
        if R_L.shape != (3, 3) or not np.all(np.isfinite(R_L)):
            raise ValueError("R_L must have shape (3, 3)")


@dataclass(frozen=True)
class RigidBodyLoadAttachmentSet:
    """Attachment geometry expressed in the payload body-fixed frame."""

    r_i_body_m: np.ndarray
    cable_lengths_m: np.ndarray

    def __post_init__(self) -> None:
        r_i_body_m = _validate_array("r_i_body_m", self.r_i_body_m, (3,))
        cable_lengths_m = np.asarray(self.cable_lengths_m, dtype=float).reshape(-1)
        if cable_lengths_m.ndim != 1 or cable_lengths_m.size != r_i_body_m.shape[0]:
            raise ValueError("cable_lengths_m must be a 1D array aligned with r_i_body_m")
        if np.any(cable_lengths_m <= 0.0) or not np.all(np.isfinite(cable_lengths_m)):
            raise ValueError("cable_lengths_m must be finite and positive")

    @property
    def count(self) -> int:
        return int(np.asarray(self.cable_lengths_m).size)

    def quadrotor_positions(self, kinematics: AttachmentKinematicsInput) -> np.ndarray:
        """Recover quadrotor positions from the paper's equation x_i = x_L + R_L(r_i - L_i q_i)."""

        if kinematics.q_body.shape[0] != self.count:
            raise ValueError("q_body count must match the attachment count")
        q_norm = np.linalg.norm(kinematics.q_body, axis=1)
        if not np.allclose(q_norm, 1.0, atol=1e-6):
            raise ValueError("each q_i must be a unit vector")

        relative_body = np.asarray(self.r_i_body_m, dtype=float) - self.cable_lengths_m[:, None] * kinematics.q_body
        return kinematics.x_L[None, :] + (kinematics.R_L @ relative_body.T).T

    def wrench_map_body(self, q_body: np.ndarray) -> np.ndarray:
        """Build the load wrench mapping Phi in the body frame for scalar tensions T."""

        q_body = _validate_array("q_body", q_body, (3,))
        if q_body.shape[0] != self.count:
            raise ValueError("q_body count must match the attachment count")
        q_norm = np.linalg.norm(q_body, axis=1)
        if not np.allclose(q_norm, 1.0, atol=1e-6):
            raise ValueError("each q_i must be a unit vector")

        phi = np.zeros((6, self.count), dtype=float)
        for idx in range(self.count):
            q_i = q_body[idx]
            r_i = np.asarray(self.r_i_body_m[idx], dtype=float)
            phi[:3, idx] = -q_i
            phi[3:, idx] = np.cross(r_i, -q_i)
        return phi


def build_three_uav_box_attachment_set(size_xyz_m: np.ndarray) -> RigidBodyLoadAttachmentSet:
    """Build a 3-point attachment layout on a box.

    The convention follows the user's current design choice:
    - two bottom-edge vertices,
    - one top-edge midpoint,
    all expressed in the payload body frame centered at the load COM.
    """

    lx, ly, lz = np.asarray(size_xyz_m, dtype=float).reshape(-1)
    half_x = lx / 2.0
    half_y = ly / 2.0
    half_z = lz / 2.0
    r_i_body_m = np.array(
        [
            [-half_x, 0.0, -half_z],
            [half_x, 0.0, -half_z],
            [0.0, 0.0, half_z],
        ],
        dtype=float,
    )
    return RigidBodyLoadAttachmentSet(
        r_i_body_m=r_i_body_m,
        cable_lengths_m=np.ones(3, dtype=float),
    )
