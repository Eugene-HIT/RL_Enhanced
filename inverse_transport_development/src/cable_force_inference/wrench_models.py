# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-25
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-25
# 目的 / Purpose:
#   实现第一版载荷 wrench 推断：先支持仅平动的所需合力计算，并保留后续
#   姿态/力矩扩展接口。
#   Implement first-stage payload wrench inference with translational force
#   recovery and leave extension points for rotational torque inference.
# 主要输入 / Main Inputs:
#   带时间戳的载荷轨迹、载荷质量、重力、可选已知外力。
#   Time-aligned payload trajectory, payload mass, gravity, and optional known
#   external forces.
# 主要输出 / Main Outputs:
#   WrenchSeries: 每个时刻的所需力与力矩。
#   WrenchSeries: required force and torque at each time sample.
# ==============================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from ..common.trajectory import TrajectorySamples3D
except ImportError:
    from common.trajectory import TrajectorySamples3D


@dataclass(frozen=True)
class PayloadGeometry:
    """Payload geometry placeholder for torque-stage modeling."""

    shape_type: str = "box"
    size_xyz_m: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.shape_type not in {"box", "custom"}:
            raise ValueError("shape_type must be 'box' or 'custom'")
        if self.size_xyz_m is not None:
            size_xyz_m = np.asarray(self.size_xyz_m, dtype=float).reshape(-1)
            if size_xyz_m.shape != (3,) or not np.all(np.isfinite(size_xyz_m)) or np.any(size_xyz_m <= 0.0):
                raise ValueError("size_xyz_m must be a positive finite 3-vector")


def inertia_matrix_for_box(mass_kg: float, size_xyz_m: np.ndarray) -> np.ndarray:
    """Return the principal inertia matrix for a uniform box about its center of mass."""

    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    lx, ly, lz = np.asarray(size_xyz_m, dtype=float).reshape(-1)
    i_xx = mass_kg * (ly * ly + lz * lz) / 12.0
    i_yy = mass_kg * (lx * lx + lz * lz) / 12.0
    i_zz = mass_kg * (lx * lx + ly * ly) / 12.0
    return np.diag([i_xx, i_yy, i_zz]).astype(float)


@dataclass(frozen=True)
class PayloadPhysicalParams:
    """Physical parameters required by the payload inverse model."""

    mass_kg: float
    gravity_mps2: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))
    inertia_kgm2: Optional[np.ndarray] = None
    geometry: Optional[PayloadGeometry] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        gravity = np.asarray(self.gravity_mps2, dtype=float).reshape(-1)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_mps2 must be a finite 3-vector")
        inertia = self.resolved_inertia_kgm2
        if inertia is not None and (inertia.shape != (3, 3) or not np.all(np.isfinite(inertia))):
            raise ValueError("resolved inertia must have shape (3, 3)")

    @property
    def resolved_inertia_kgm2(self) -> Optional[np.ndarray]:
        if self.inertia_kgm2 is not None:
            return np.asarray(self.inertia_kgm2, dtype=float)
        if self.geometry is not None and self.geometry.shape_type == "box" and self.geometry.size_xyz_m is not None:
            return inertia_matrix_for_box(self.mass_kg, self.geometry.size_xyz_m)
        return None


@dataclass(frozen=True)
class WrenchSeries:
    """Time-aligned required payload wrench."""

    time_s: np.ndarray
    force_n: np.ndarray
    torque_nm: np.ndarray
    source: str

    def __post_init__(self) -> None:
        time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        force_n = np.asarray(self.force_n, dtype=float)
        torque_nm = np.asarray(self.torque_nm, dtype=float)
        if force_n.ndim != 2 or force_n.shape[1] != 3:
            raise ValueError("force_n must have shape (N, 3)")
        if torque_nm.ndim != 2 or torque_nm.shape[1] != 3:
            raise ValueError("torque_nm must have shape (N, 3)")
        if force_n.shape[0] != time_s.size or torque_nm.shape[0] != time_s.size:
            raise ValueError("time_s, force_n, and torque_nm must be aligned")
        if not np.all(np.isfinite(force_n)) or not np.all(np.isfinite(torque_nm)):
            raise ValueError("force_n and torque_nm must be finite")


def compute_translational_wrench(
    trajectory: TrajectorySamples3D,
    params: PayloadPhysicalParams,
    known_external_force_n: Optional[np.ndarray] = None,
) -> WrenchSeries:
    """Compute the applied payload wrench required to realize the trajectory.

    The current stage only models translational dynamics. Required torque is set
    to zero and must be replaced by a rotational model once payload attitude and
    inertia are promoted to first-class inputs.
    """

    time_s = np.asarray(trajectory.time_s, dtype=float)
    acceleration_mps2 = np.asarray(trajectory.acceleration_mps2, dtype=float)
    gravity_mps2 = np.asarray(params.gravity_mps2, dtype=float).reshape(1, 3)

    if known_external_force_n is None:
        known_external_force_n = np.zeros_like(acceleration_mps2)
    else:
        known_external_force_n = np.asarray(known_external_force_n, dtype=float)
        if known_external_force_n.shape != acceleration_mps2.shape:
            raise ValueError("known_external_force_n must match acceleration shape")

    # Newton's law with gravity convention carried explicitly in gravity_mps2.
    required_force_n = params.mass_kg * (acceleration_mps2 - gravity_mps2) - known_external_force_n
    required_torque_nm = np.zeros_like(required_force_n)

    return WrenchSeries(
        time_s=time_s,
        force_n=required_force_n,
        torque_nm=required_torque_nm,
        source=f"translational::{trajectory.source}",
    )


def compute_payload_wrench(
    trajectory: TrajectorySamples3D,
    params: PayloadPhysicalParams,
    known_external_force_n: Optional[np.ndarray] = None,
    known_external_torque_nm: Optional[np.ndarray] = None,
) -> WrenchSeries:
    """Compute full payload wrench with translational force and rigid-body torque.

    At this stage torque recovery only depends on payload angular states and inertia.
    Cable attachment points and force application geometry remain intentionally out of scope.
    """

    translational = compute_translational_wrench(
        trajectory=trajectory,
        params=params,
        known_external_force_n=known_external_force_n,
    )

    inertia_kgm2 = params.resolved_inertia_kgm2
    if inertia_kgm2 is None:
        torque_nm = np.zeros_like(translational.force_n)
    else:
        if trajectory.angular_velocity_radps is None or trajectory.angular_acceleration_radps2 is None:
            raise ValueError(
                "angular_velocity_radps and angular_acceleration_radps2 are required when inertia is provided"
            )
        angular_velocity_radps = np.asarray(trajectory.angular_velocity_radps, dtype=float)
        angular_acceleration_radps2 = np.asarray(trajectory.angular_acceleration_radps2, dtype=float)
        inertia_times_omega = angular_velocity_radps @ inertia_kgm2.T
        inertia_times_alpha = angular_acceleration_radps2 @ inertia_kgm2.T
        gyroscopic = np.cross(angular_velocity_radps, inertia_times_omega)
        torque_nm = inertia_times_alpha + gyroscopic

    if known_external_torque_nm is None:
        known_external_torque_nm = np.zeros_like(torque_nm)
    else:
        known_external_torque_nm = np.asarray(known_external_torque_nm, dtype=float)
        if known_external_torque_nm.shape != torque_nm.shape:
            raise ValueError("known_external_torque_nm must match torque shape")

    return WrenchSeries(
        time_s=np.asarray(trajectory.time_s, dtype=float),
        force_n=translational.force_n,
        torque_nm=torque_nm - known_external_torque_nm,
        source=f"full::{trajectory.source}",
    )
