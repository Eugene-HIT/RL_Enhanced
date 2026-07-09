# -*- coding: utf-8 -*-
# ==============================================
# 创建时间 / Created: 2026-05-25
# 创建者 / Creator: Eugene
# 最后修改 / Last Modified: 2026-05-25
# 目的 / Purpose:
#   定义逆运输阶段的统一轨迹输入模型，并支持从位置样本或分段多项式轨迹
#   生成带速度/加速度的时间序列。
#   Define canonical trajectory input models for inverse transport and support
#   derivative generation from either position samples or piecewise polynomials.
# 主要输入 / Main Inputs:
#   时间戳、位置样本、可选导数；或分段多项式系数与段时长。
#   Time stamps, position samples, optional derivatives; or polynomial
#   coefficients with per-segment durations.
# 主要输出 / Main Outputs:
#   TrajectorySamples3D: 统一后的时序轨迹。
#   PiecewisePolynomialTrajectory3D: 可解析求导的轨迹表示。
# ==============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _validate_time_vector(time_s: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be a 1D array with at least two samples")
    dt = np.diff(time_s)
    if not np.all(np.isfinite(time_s)) or np.any(dt <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    return time_s


def _validate_matrix(name: str, value: np.ndarray, cols: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != cols:
        raise ValueError(f"{name} must have shape (N, {cols})")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_difference_nonuniform(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    derivatives = np.empty_like(values, dtype=float)
    for axis in range(values.shape[1]):
        derivatives[:, axis] = np.gradient(values[:, axis], time_s, edge_order=2)
    return derivatives


def _poly_basis(t: float, order: int, deriv: int) -> np.ndarray:
    basis = np.zeros(order + 1, dtype=float)
    for power in range(order + 1):
        if power < deriv:
            continue
        coeff = 1.0
        for offset in range(deriv):
            coeff *= power - offset
        basis[power] = coeff * (t ** (power - deriv))
    return basis


@dataclass(frozen=True)
class TrajectorySamples3D:
    """Canonical time-aligned payload trajectory samples for inverse inference."""

    time_s: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    acceleration_mps2: np.ndarray
    orientation_rpy_rad: Optional[np.ndarray] = None
    angular_velocity_radps: Optional[np.ndarray] = None
    angular_acceleration_radps2: Optional[np.ndarray] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        time_s = _validate_time_vector(self.time_s)
        position_m = _validate_matrix("position_m", self.position_m, 3)
        velocity_mps = _validate_matrix("velocity_mps", self.velocity_mps, 3)
        acceleration_mps2 = _validate_matrix("acceleration_mps2", self.acceleration_mps2, 3)
        if position_m.shape[0] != time_s.size:
            raise ValueError("position_m row count must match time_s length")
        if velocity_mps.shape[0] != time_s.size:
            raise ValueError("velocity_mps row count must match time_s length")
        if acceleration_mps2.shape[0] != time_s.size:
            raise ValueError("acceleration_mps2 row count must match time_s length")

        if self.orientation_rpy_rad is not None:
            orientation = _validate_matrix("orientation_rpy_rad", self.orientation_rpy_rad, 3)
            if orientation.shape[0] != time_s.size:
                raise ValueError("orientation_rpy_rad row count must match time_s length")
        if self.angular_velocity_radps is not None:
            ang_vel = _validate_matrix("angular_velocity_radps", self.angular_velocity_radps, 3)
            if ang_vel.shape[0] != time_s.size:
                raise ValueError("angular_velocity_radps row count must match time_s length")
        if self.angular_acceleration_radps2 is not None:
            ang_acc = _validate_matrix("angular_acceleration_radps2", self.angular_acceleration_radps2, 3)
            if ang_acc.shape[0] != time_s.size:
                raise ValueError("angular_acceleration_radps2 row count must match time_s length")

    @classmethod
    def from_position_samples(
        cls,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_mps: Optional[np.ndarray] = None,
        acceleration_mps2: Optional[np.ndarray] = None,
        orientation_rpy_rad: Optional[np.ndarray] = None,
        angular_velocity_radps: Optional[np.ndarray] = None,
        angular_acceleration_radps2: Optional[np.ndarray] = None,
        source: str = "position_samples",
    ) -> "TrajectorySamples3D":
        time_s = _validate_time_vector(time_s)
        position_m = _validate_matrix("position_m", position_m, 3)
        if position_m.shape[0] != time_s.size:
            raise ValueError("position_m row count must match time_s length")

        if velocity_mps is None:
            velocity_mps = _finite_difference_nonuniform(time_s, position_m)
        else:
            velocity_mps = _validate_matrix("velocity_mps", velocity_mps, 3)

        if acceleration_mps2 is None:
            acceleration_mps2 = _finite_difference_nonuniform(time_s, velocity_mps)
        else:
            acceleration_mps2 = _validate_matrix("acceleration_mps2", acceleration_mps2, 3)

        if orientation_rpy_rad is not None:
            orientation_rpy_rad = _validate_matrix("orientation_rpy_rad", orientation_rpy_rad, 3)
            if orientation_rpy_rad.shape[0] != time_s.size:
                raise ValueError("orientation_rpy_rad row count must match time_s length")

            if angular_velocity_radps is None:
                angular_velocity_radps = _finite_difference_nonuniform(time_s, orientation_rpy_rad)
            else:
                angular_velocity_radps = _validate_matrix("angular_velocity_radps", angular_velocity_radps, 3)

            if angular_acceleration_radps2 is None:
                angular_acceleration_radps2 = _finite_difference_nonuniform(time_s, angular_velocity_radps)
            else:
                angular_acceleration_radps2 = _validate_matrix("angular_acceleration_radps2", angular_acceleration_radps2, 3)
        else:
            if angular_velocity_radps is not None or angular_acceleration_radps2 is not None:
                raise ValueError("orientation_rpy_rad is required when angular derivatives are provided")

        return cls(
            time_s=time_s,
            position_m=position_m,
            velocity_mps=velocity_mps,
            acceleration_mps2=acceleration_mps2,
            orientation_rpy_rad=orientation_rpy_rad,
            angular_velocity_radps=angular_velocity_radps,
            angular_acceleration_radps2=angular_acceleration_radps2,
            source=source,
        )


@dataclass(frozen=True)
class PiecewisePolynomialTrajectory3D:
    """Piecewise polynomial trajectory that supports analytic derivatives."""

    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    t_per_seg: np.ndarray
    order: int
    source: str = "piecewise_polynomial"

    def __post_init__(self) -> None:
        coeffs_x = _validate_matrix("coeffs_x", self.coeffs_x, self.order + 1)
        coeffs_y = _validate_matrix("coeffs_y", self.coeffs_y, self.order + 1)
        coeffs_z = _validate_matrix("coeffs_z", self.coeffs_z, self.order + 1)
        t_per_seg = np.asarray(self.t_per_seg, dtype=float).reshape(-1)
        if coeffs_x.shape != coeffs_y.shape or coeffs_x.shape != coeffs_z.shape:
            raise ValueError("coeff matrices must have identical shapes")
        if coeffs_x.shape[0] != t_per_seg.size:
            raise ValueError("number of segments must match t_per_seg length")
        if np.any(t_per_seg <= 0.0) or not np.all(np.isfinite(t_per_seg)):
            raise ValueError("t_per_seg must be finite and positive")

    def sample(self, dt: float = 0.08, include_segment_end: bool = True) -> TrajectorySamples3D:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        sample_t = []
        position = []
        velocity = []
        acceleration = []
        t_global = 0.0

        for seg_idx, duration in enumerate(np.asarray(self.t_per_seg, dtype=float)):
            count = max(2, int(np.ceil(duration / dt)))
            ts_local = np.linspace(0.0, duration, count)
            if not include_segment_end and seg_idx < len(self.t_per_seg) - 1:
                ts_local = ts_local[:-1]

            for t_local in ts_local:
                phi0 = _poly_basis(float(t_local), order=self.order, deriv=0)
                phi1 = _poly_basis(float(t_local), order=self.order, deriv=1)
                phi2 = _poly_basis(float(t_local), order=self.order, deriv=2)

                px = float(phi0 @ self.coeffs_x[seg_idx])
                py = float(phi0 @ self.coeffs_y[seg_idx])
                pz = float(phi0 @ self.coeffs_z[seg_idx])
                vx = float(phi1 @ self.coeffs_x[seg_idx])
                vy = float(phi1 @ self.coeffs_y[seg_idx])
                vz = float(phi1 @ self.coeffs_z[seg_idx])
                ax = float(phi2 @ self.coeffs_x[seg_idx])
                ay = float(phi2 @ self.coeffs_y[seg_idx])
                az = float(phi2 @ self.coeffs_z[seg_idx])

                sample_t.append(t_global + float(t_local))
                position.append([px, py, pz])
                velocity.append([vx, vy, vz])
                acceleration.append([ax, ay, az])

            t_global += float(duration)

        time_s = np.asarray(sample_t, dtype=float)
        position_m = np.asarray(position, dtype=float)
        velocity_mps = np.asarray(velocity, dtype=float)
        acceleration_mps2 = np.asarray(acceleration, dtype=float)

        _, unique_idx = np.unique(np.round(time_s, decimals=12), return_index=True)
        unique_idx = np.sort(unique_idx)
        return TrajectorySamples3D(
            time_s=time_s[unique_idx],
            position_m=position_m[unique_idx],
            velocity_mps=velocity_mps[unique_idx],
            acceleration_mps2=acceleration_mps2[unique_idx],
            source=self.source,
        )
