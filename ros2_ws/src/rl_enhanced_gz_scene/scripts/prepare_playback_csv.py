#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def sample_theta_from_export(
    pos: np.ndarray,
    seg_id: int,
    hard_idx: np.ndarray,
    p_wp: np.ndarray,
    roll_wp: np.ndarray,
) -> float:
    seg_id = int(np.clip(seg_id, 0, len(hard_idx) - 2))
    ia = int(hard_idx[seg_id])
    ib = int(hard_idx[seg_id + 1])
    a = p_wp[ia]
    b = p_wp[ib]
    ra = float(roll_wp[ia])
    rb = float(roll_wp[ib])
    delta = b - a
    denom = float(np.dot(delta, delta))
    if denom < 1e-9:
        return ra
    alpha = float(np.clip(np.dot(pos - a, delta) / denom, 0.0, 1.0))
    return ra + alpha * wrap_to_pi(rb - ra)


def load_export_attitude(export_path: str) -> dict[str, np.ndarray] | None:
    try:
        mat = loadmat(export_path, squeeze_me=True, struct_as_record=False)
        traj = mat["traj"]
        key = mat["keyframes"]
        sample_seg = np.asarray(getattr(traj, "sample_seg"), dtype=int).reshape(-1)
        hard_idx = np.asarray(getattr(traj, "hard_idx"), dtype=int).reshape(-1)
        p_wp = np.asarray(getattr(key, "P_wp"), dtype=float)
        roll_wp = np.asarray(getattr(key, "roll_wp"), dtype=float).reshape(-1)
    except Exception:
        return None

    if p_wp.ndim != 2 or p_wp.shape[1] != 3 or roll_wp.ndim != 1:
        return None
    if hard_idx.ndim != 1 or len(hard_idx) < 2 or len(sample_seg) < 1:
        return None
    if int(np.max(hard_idx)) >= p_wp.shape[0] or int(np.max(hard_idx)) >= roll_wp.shape[0]:
        return None

    return {
        "sample_seg": sample_seg,
        "hard_idx": hard_idx,
        "p_wp": p_wp,
        "roll_wp": roll_wp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--time", dest="time_path", required=True)
    parser.add_argument("--export", default="/home/eugene/RL_enhanced/corridor_export.mat")
    parser.add_argument("--out", required=True)
    parser.add_argument("--theta-sign", type=float, default=1.0)
    parser.add_argument("--flip-theta-ymin", type=float, default=float("nan"))
    parser.add_argument("--flip-theta-ymax", type=float, default=float("nan"))
    parser.add_argument("--flip-theta-sign", type=float, default=-1.0)
    args = parser.parse_args()

    xyz = np.load(args.xyz).astype(float)
    t = np.load(args.time_path).astype(float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or t.ndim != 1 or xyz.shape[0] != t.shape[0]:
        raise ValueError("invalid trajectory shapes")

    attitude = load_export_attitude(args.export) if args.export else None
    flip_enabled = np.isfinite(args.flip_theta_ymin) and np.isfinite(args.flip_theta_ymax)
    if flip_enabled:
        y_lo = min(args.flip_theta_ymin, args.flip_theta_ymax)
        y_hi = max(args.flip_theta_ymin, args.flip_theta_ymax)
    else:
        y_lo, y_hi = 0.0, -1.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# t,x,y,z,theta\n")
        for i in range(xyz.shape[0]):
            pos = xyz[i]
            theta = 0.0
            if attitude is not None:
                seg = int(attitude["sample_seg"][min(i, len(attitude["sample_seg"]) - 1)])
                theta = sample_theta_from_export(
                    pos,
                    seg,
                    attitude["hard_idx"],
                    attitude["p_wp"],
                    attitude["roll_wp"],
                )
                theta = args.theta_sign * theta
                if flip_enabled and (y_lo <= float(pos[1]) <= y_hi):
                    theta = args.flip_theta_sign * theta
            f.write(
                f"{float(t[i]):.9f},{float(pos[0]):.9f},{float(pos[1]):.9f},"
                f"{float(pos[2]):.9f},{float(theta):.9f}\n"
            )

    print(f"[prepare_playback_csv] wrote {xyz.shape[0]} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
