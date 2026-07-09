# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-26
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-26
# 目的 / Purpose:
#   在尚未固定绳索方向 q_i 之前，先把载荷总 wrench 分配到各挂点的三维受力
#   f_i 上，为后续张力/方向分解提供中间层。
#   Allocate the total payload wrench to 3D forces applied at attachment points
#   before cable directions q_i are fixed, providing an intermediate layer for
#   later tension/direction decomposition.
# 主要输入 / Main Inputs:
#   载荷 wrench、挂点位置、可选正则权重。
#   Payload wrench, attachment positions, and optional regularization weights.
# 主要输出 / Main Outputs:
#   每个挂点的三维力、重构误差、以及后续可直接转成 q_i / T_i 的中间结果。
#   Per-attachment 3D forces, reconstruction error, and an intermediate result
#   that can later be converted into q_i / T_i.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ..common.rigid_body_payload import (
        AttachmentKinematicsInput,
        RigidBodyLoadAttachmentSet,
        rotation_matrix_from_rpy,
        skew_symmetric,
    )
except ImportError:
    from common.rigid_body_payload import (
        AttachmentKinematicsInput,
        RigidBodyLoadAttachmentSet,
        rotation_matrix_from_rpy,
        skew_symmetric,
    )

try:
    from .wrench_models import WrenchSeries
except ImportError:
    from wrench_models import WrenchSeries

try:
    from ..common.trajectory import TrajectorySamples3D
except ImportError:
    from common.trajectory import TrajectorySamples3D


@dataclass(frozen=True)
class AttachmentForceAllocationResult:
    """Force allocation result on payload attachment points."""

    attachment_force_body_n: np.ndarray
    reconstructed_wrench_body: np.ndarray
    residual_wrench_body: np.ndarray


@dataclass(frozen=True)
class CableForceInferenceSeries:
    """Time series of per-attachment forces, cable directions, and UAV positions."""

    time_s: np.ndarray
    payload_wrench_body: np.ndarray
    attachment_force_body_n: np.ndarray
    q_body: np.ndarray
    force_norm_n: np.ndarray
    quadrotor_position_inertial_m: np.ndarray
    residual_wrench_body: np.ndarray


def build_attachment_wrench_matrix(attachments: RigidBodyLoadAttachmentSet) -> np.ndarray:
    """Build the 6 x (3n) matrix G such that G f = W for stacked attachment forces."""

    n_attach = attachments.count
    matrix = np.zeros((6, 3 * n_attach), dtype=float)
    for idx in range(n_attach):
        r_i = np.asarray(attachments.r_i_body_m[idx], dtype=float)
        col = slice(3 * idx, 3 * (idx + 1))
        matrix[:3, col] = np.eye(3, dtype=float)
        matrix[3:, col] = skew_symmetric(r_i)
    return matrix


def allocate_attachment_forces(
    wrench_body: np.ndarray,
    attachments: RigidBodyLoadAttachmentSet,
    ridge: float = 1e-9,
) -> AttachmentForceAllocationResult:
    """Allocate a body-frame payload wrench to 3D forces at each attachment point.

    This stage intentionally solves for vector forces f_i instead of scalar cable
    tensions T_i. Later, once cable directions are modeled, each f_i can be
    decomposed into T_i = ||f_i|| and q_i = -f_i / ||f_i|| under pull-only cable
    assumptions.
    """

    wrench_body = np.asarray(wrench_body, dtype=float).reshape(-1)
    if wrench_body.shape != (6,) or not np.all(np.isfinite(wrench_body)):
        raise ValueError("wrench_body must be a finite 6-vector")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    matrix = build_attachment_wrench_matrix(attachments)
    gram = matrix @ matrix.T
    if ridge > 0.0:
        gram = gram + ridge * np.eye(6, dtype=float)

    dual = np.linalg.solve(gram, wrench_body)
    stacked_force = matrix.T @ dual
    reconstructed = matrix @ stacked_force
    residual = wrench_body - reconstructed

    return AttachmentForceAllocationResult(
        attachment_force_body_n=stacked_force.reshape(attachments.count, 3),
        reconstructed_wrench_body=reconstructed,
        residual_wrench_body=residual,
    )


def forces_to_cable_state(attachment_force_body_n: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Convert attachment forces to cable directions and scalar tensions."""

    attachment_force_body_n = np.asarray(attachment_force_body_n, dtype=float)
    if attachment_force_body_n.ndim != 2 or attachment_force_body_n.shape[1] != 3:
        raise ValueError("attachment_force_body_n must have shape (N, 3)")

    tensions_n = np.linalg.norm(attachment_force_body_n, axis=1)
    q_body = np.zeros_like(attachment_force_body_n)
    active = tensions_n > eps
    q_body[active] = -attachment_force_body_n[active] / tensions_n[active, None]
    return q_body, tensions_n


def infer_cable_force_series(
    trajectory: TrajectorySamples3D,
    wrench_series: WrenchSeries,
    attachments: RigidBodyLoadAttachmentSet,
    ridge: float = 1e-9,
) -> CableForceInferenceSeries:
    """Infer per-attachment forces and UAV positions over time.

    Force is transformed from the inertial frame to the payload body frame using
    the payload orientation, while torque is assumed to already be represented in
    the body frame as produced by the current rigid-body wrench model.
    """

    time_s = np.asarray(trajectory.time_s, dtype=float)
    if wrench_series.time_s.shape[0] != time_s.shape[0]:
        raise ValueError("trajectory and wrench_series must have the same sample count")

    n_sample = time_s.size
    n_attach = attachments.count
    payload_wrench_body = np.zeros((n_sample, 6), dtype=float)
    attachment_force_body_n = np.zeros((n_sample, n_attach, 3), dtype=float)
    q_body = np.zeros_like(attachment_force_body_n)
    force_norm_n = np.zeros((n_sample, n_attach), dtype=float)
    quadrotor_position_inertial_m = np.zeros((n_sample, n_attach, 3), dtype=float)
    residual_wrench_body = np.zeros((n_sample, 6), dtype=float)

    for idx in range(n_sample):
        if trajectory.orientation_rpy_rad is None:
            rotation = np.eye(3, dtype=float)
        else:
            rotation = rotation_matrix_from_rpy(trajectory.orientation_rpy_rad[idx])

        force_body = rotation.T @ np.asarray(wrench_series.force_n[idx], dtype=float)
        torque_body = np.asarray(wrench_series.torque_nm[idx], dtype=float)
        wrench_body = np.concatenate([force_body, torque_body], axis=0)

        allocation = allocate_attachment_forces(wrench_body=wrench_body, attachments=attachments, ridge=ridge)
        q_now, norm_now = forces_to_cable_state(allocation.attachment_force_body_n)
        positions = attachments.quadrotor_positions(
            AttachmentKinematicsInput(
                x_L=np.asarray(trajectory.position_m[idx], dtype=float),
                R_L=rotation,
                q_body=q_now,
            )
        )

        payload_wrench_body[idx] = wrench_body
        attachment_force_body_n[idx] = allocation.attachment_force_body_n
        q_body[idx] = q_now
        force_norm_n[idx] = norm_now
        quadrotor_position_inertial_m[idx] = positions
        residual_wrench_body[idx] = allocation.residual_wrench_body

    return CableForceInferenceSeries(
        time_s=time_s,
        payload_wrench_body=payload_wrench_body,
        attachment_force_body_n=attachment_force_body_n,
        q_body=q_body,
        force_norm_n=force_norm_n,
        quadrotor_position_inertial_m=quadrotor_position_inertial_m,
        residual_wrench_body=residual_wrench_body,
    )