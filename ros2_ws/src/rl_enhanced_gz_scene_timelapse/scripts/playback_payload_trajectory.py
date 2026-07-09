#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quat_from_pitch_y(theta: float) -> tuple[float, float, float, float]:
    half = 0.5 * theta
    return (0.0, math.sin(half), 0.0, math.cos(half))


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def quat_rotate_vec(quat: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    u = np.array([qx, qy, qz], dtype=float)
    s = float(qw)
    uv = float(np.dot(u, v))
    uu = float(np.dot(u, u))
    return 2.0 * uv * u + (s * s - uu) * v + 2.0 * s * np.cross(u, v)


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
    except Exception as exc:
        print(f"[playback] warning: failed to load export attitude from {export_path}: {exc}")
        return None

    if p_wp.ndim != 2 or p_wp.shape[1] != 3 or roll_wp.ndim != 1:
        print("[playback] warning: export attitude arrays are invalid")
        return None
    if hard_idx.ndim != 1 or len(hard_idx) < 2 or len(sample_seg) < 1:
        print("[playback] warning: export segment arrays are invalid")
        return None
    if int(np.max(hard_idx)) >= p_wp.shape[0] or int(np.max(hard_idx)) >= roll_wp.shape[0]:
        print("[playback] warning: export hard_idx out of bounds")
        return None

    return {
        "sample_seg": sample_seg,
        "hard_idx": hard_idx,
        "p_wp": p_wp,
        "roll_wp": roll_wp,
    }


def call_set_pose(
    world: str,
    entity: str,
    pos: np.ndarray,
    quat: tuple[float, float, float, float],
    timeout_ms: int = 300,
) -> bool:
    qx, qy, qz, qw = quat
    req = (
        f'name: "{entity}", '
        f'position: {{x: {pos[0]:.6f}, y: {pos[1]:.6f}, z: {pos[2]:.6f}}}, '
        f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}'
    )
    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{world}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(max(50, int(timeout_ms))),
            "--req",
            req,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "data: true" in (result.stdout + result.stderr)


def spawn_trail_segment(world: str, name: str, p0: np.ndarray, p1: np.ndarray, thickness: float) -> None:
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-4:
        return

    center = 0.5 * (p0 + p1)
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    horiz = math.hypot(float(delta[0]), float(delta[1]))
    pitch = -math.atan2(float(delta[2]), horiz)

    model = (
        '<sdf version="1.10">'
        f'<model name="{name}">'
        '<static>true</static>'
        f'<pose>{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 0 {pitch:.6f} {yaw:.6f}</pose>'
        '<link name="link">'
        '<visual name="visual">'
        f'<geometry><box><size>{length:.6f} {thickness:.6f} {thickness:.6f}</size></box></geometry>'
        '<material>'
        '<ambient>0.98 0.82 0.18 1</ambient>'
        '<diffuse>0.98 0.82 0.18 1</diffuse>'
        '<emissive>0.10 0.08 0.02 1</emissive>'
        '</material>'
        '</visual>'
        '</link>'
        '</model>'
        '</sdf>'
    )

    subprocess.run(
        [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            world,
            "-string",
            model,
            "-name",
            name,
            "-allow_renaming",
            "true",
        ],
        capture_output=True,
        text=True,
    )


def spawn_model_string(
    world: str,
    name: str,
    model_sdf: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
) -> bool:
    result = subprocess.run(
        [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            world,
            "-string",
            model_sdf,
            "-name",
            name,
            "-allow_renaming",
            "false",
            "-x",
            f"{x:.6f}",
            "-y",
            f"{y:.6f}",
            "-z",
            f"{z:.6f}",
            "-Y",
            f"{yaw:.6f}",
        ],
        capture_output=True,
        text=True,
    )
    out = (result.stdout + result.stderr).lower()
    if result.returncode != 0:
        return False
    if "error" in out or "failed" in out:
        return False
    return True


def make_drone_model_sdf(name: str, mesh_uri: str, scale: float) -> str:
    return (
        '<sdf version="1.10">'
        f'<model name="{name}">'
        "<static>false</static>"
        '<link name="link">'
        "<gravity>false</gravity>"
        "<inertial>"
        "<mass>0.15</mass>"
        "<inertia><ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0008</izz></inertia>"
        "</inertial>"
        '<visual name="visual">'
        "<geometry><box><size>0.30 0.30 0.09</size></box></geometry>"
        "<material>"
        "<ambient>0.96 0.84 0.18 1</ambient>"
        "<diffuse>0.96 0.84 0.18 1</diffuse>"
        "<emissive>0.20 0.14 0.02 1</emissive>"
        "</material>"
        "</visual>"
        '<visual name="arm_x">'
        "<pose>0 0 0 0 1.570796 0</pose>"
        "<geometry><cylinder><radius>0.018</radius><length>0.55</length></cylinder></geometry>"
        "<material><ambient>0.22 0.22 0.22 1</ambient><diffuse>0.22 0.22 0.22 1</diffuse></material>"
        "</visual>"
        '<visual name="arm_y">'
        "<pose>0 0 0 0 1.570796 1.570796</pose>"
        "<geometry><cylinder><radius>0.018</radius><length>0.55</length></cylinder></geometry>"
        "<material><ambient>0.22 0.22 0.22 1</ambient><diffuse>0.22 0.22 0.22 1</diffuse></material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


def make_tether_model_sdf(name: str, radius: float, length: float) -> str:
    return (
        '<sdf version="1.10">'
        f'<model name="{name}">'
        "<static>false</static>"
        '<link name="link">'
        "<gravity>false</gravity>"
        "<inertial>"
        "<mass>0.03</mass>"
        "<inertia><ixx>0.00002</ixx><iyy>0.00002</iyy><izz>0.00002</izz></inertia>"
        "</inertial>"
        '<visual name="visual">'
        "<geometry><cylinder>"
        f"<radius>{radius:.6f}</radius>"
        f"<length>{length:.6f}</length>"
        "</cylinder></geometry>"
        "<material>"
        "<ambient>0.95 0.92 0.78 1</ambient>"
        "<diffuse>0.95 0.92 0.78 1</diffuse>"
        "<emissive>0.08 0.08 0.05 1</emissive>"
        "</material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


def make_drone_fallback_box_sdf(name: str) -> str:
    return (
        '<sdf version="1.10">'
        f'<model name="{name}">'
        "<static>true</static>"
        '<link name="link">'
        '<visual name="body">'
        "<geometry><box><size>0.16 0.16 0.05</size></box></geometry>"
        "<material>"
        "<ambient>0.05 0.62 0.95 1</ambient>"
        "<diffuse>0.05 0.62 0.95 1</diffuse>"
        "</material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


def make_rope_node_sdf(name: str, radius: float) -> str:
    return (
        '<sdf version="1.10">'
        f'<model name="{name}">'
        "<static>false</static>"
        '<link name="link">'
        "<gravity>false</gravity>"
        "<inertial>"
        "<mass>0.01</mass>"
        "<inertia><ixx>0.00001</ixx><iyy>0.00001</iyy><izz>0.00001</izz></inertia>"
        "</inertial>"
        '<visual name="visual">'
        "<geometry><sphere>"
        f"<radius>{radius:.6f}</radius>"
        "</sphere></geometry>"
        "<material>"
        "<ambient>0.95 0.92 0.78 1</ambient>"
        "<diffuse>0.95 0.92 0.78 1</diffuse>"
        "<emissive>0.08 0.08 0.05 1</emissive>"
        "</material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


def interpolate_pose(elapsed: float, t: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    if elapsed <= float(t[0]):
        p = xyz[0]
        d = xyz[min(1, len(xyz) - 1)] - xyz[0]
    elif elapsed >= float(t[-1]):
        p = xyz[-1]
        d = xyz[-1] - xyz[max(len(xyz) - 2, 0)]
    else:
        idx = int(np.searchsorted(t, elapsed, side="right"))
        i0 = max(0, idx - 1)
        i1 = min(len(t) - 1, idx)
        alpha = (elapsed - float(t[i0])) / max(float(t[i1] - t[i0]), 1e-9)
        p = (1.0 - alpha) * xyz[i0] + alpha * xyz[i1]
        d = xyz[i1] - xyz[i0]

    yaw = math.atan2(float(d[1]), float(d[0])) if np.linalg.norm(d[:2]) > 1e-9 else 0.0
    return p, yaw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--entity", default="trajectory_payload")
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--time", dest="time_path", required=True)
    parser.add_argument("--export", default="/home/eugene/RL_enhanced/corridor_export.mat")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--theta-sign", type=float, default=1.0)
    parser.add_argument("--flip-theta-ymin", type=float, default=float("nan"))
    parser.add_argument("--flip-theta-ymax", type=float, default=float("nan"))
    parser.add_argument("--flip-theta-sign", type=float, default=-1.0)
    parser.add_argument("--goal-marker-entity", default="")
    parser.add_argument("--goal-marker-x", type=float, default=0.0)
    parser.add_argument("--goal-marker-y", type=float, default=0.0)
    parser.add_argument("--goal-marker-z", type=float, default=0.0)
    parser.add_argument("--goal-marker-yaw-rate", type=float, default=0.30)
    parser.add_argument("--world-drones", default="true")
    parser.add_argument("--world-drones-spawn", default="false")
    parser.add_argument(
        "--drone-mesh-uri",
        default="file:///home/eugene/ros2_ws/src/rl_enhanced_gz_scene/models/payload_box/meshes/quadrotor_base.dae",
    )
    parser.add_argument("--drone-scale", type=float, default=0.16)
    parser.add_argument("--drone-z-rel", type=float, default=0.55)
    parser.add_argument("--drone-hook-z-rel", type=float, default=-0.06)
    parser.add_argument("--rope-enable", default="false")
    parser.add_argument("--rope-node-count", type=int, default=5)
    parser.add_argument("--rope-node-radius", type=float, default=0.013)
    parser.add_argument("--rope-sag", type=float, default=0.06)
    parser.add_argument("--rope-update-every", type=int, default=3)
    parser.add_argument("--marker-update-every", type=int, default=2)
    parser.add_argument("--service-timeout-ms", type=int, default=260)
    parser.add_argument("--trail-enable", default="false")
    parser.add_argument("--dt", type=float, default=0.14)
    parser.add_argument("--trail-period", type=float, default=0.8)
    parser.add_argument("--trail-radius", type=float, default=0.07)
    args = parser.parse_args()

    xyz = np.load(args.xyz).astype(float)
    t = np.load(args.time_path).astype(float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or t.ndim != 1 or xyz.shape[0] != t.shape[0]:
        print("Trajectory file shapes are invalid", file=sys.stderr)
        return 1

    attitude = load_export_attitude(args.export) if args.export else None
    if attitude is not None:
        print(f"[playback] loaded fixed3d attitude from {args.export}")
    else:
        print("[playback] attitude source unavailable; fallback to heading-yaw replay")

    print(
        f"[playback] loaded {xyz.shape[0]} samples, duration={float(t[-1]):.3f}s, "
        f"speed={args.speed:.2f}x, z_offset={args.z_offset:.3f}m, theta_sign={args.theta_sign:.2f}"
    )
    flip_enabled = np.isfinite(args.flip_theta_ymin) and np.isfinite(args.flip_theta_ymax)
    if flip_enabled:
        y_lo = min(args.flip_theta_ymin, args.flip_theta_ymax)
        y_hi = max(args.flip_theta_ymin, args.flip_theta_ymax)
        print(
            f"[playback] theta flip enabled in y-range [{y_lo:.3f}, {y_hi:.3f}], "
            f"sign={args.flip_theta_sign:.2f}"
        )
    else:
        y_lo = 0.0
        y_hi = -1.0
    marker_enabled = bool(args.goal_marker_entity.strip())
    marker_pos = np.array([args.goal_marker_x, args.goal_marker_y, args.goal_marker_z], dtype=float)
    marker_last_ok = True
    trail_enable = parse_bool(args.trail_enable)
    rope_enable = parse_bool(args.rope_enable)
    rope_update_every = max(1, int(args.rope_update_every))
    marker_update_every = max(1, int(args.marker_update_every))
    service_timeout_ms = max(80, int(args.service_timeout_ms))
    world_drones = parse_bool(args.world_drones)
    world_drones_spawn = parse_bool(args.world_drones_spawn)
    drone_specs = [
        {
            "drone": "world_drone_front",
            "xy_off": np.array([0.26, 0.00], dtype=float),
            "yaw_w": 0.00,
            "anchor_local": np.array([0.20, 0.00, 0.09], dtype=float),
            "rope_prefix": "world_rope_front",
        },
        {
            "drone": "world_drone_rear_left",
            "xy_off": np.array([-0.22, 0.17], dtype=float),
            "yaw_w": 0.60,
            "anchor_local": np.array([-0.18, 0.13, 0.09], dtype=float),
            "rope_prefix": "world_rope_rear_left",
        },
        {
            "drone": "world_drone_rear_right",
            "xy_off": np.array([-0.22, -0.17], dtype=float),
            "yaw_w": -0.60,
            "anchor_local": np.array([-0.18, -0.13, 0.09], dtype=float),
            "rope_prefix": "world_rope_rear_right",
        },
    ]
    drone_pose_ok: dict[str, bool] = {}
    rope_nodes: dict[str, list[str]] = {}
    if marker_enabled:
        print(
            f"[playback] goal marker '{args.goal_marker_entity}' at "
            f"({marker_pos[0]:.3f}, {marker_pos[1]:.3f}, {marker_pos[2]:.3f}), "
            f"yaw_rate={args.goal_marker_yaw_rate:.3f} rad/s"
        )
    print(
        f"[playback] perf: dt={args.dt:.2f}s, rope_nodes={max(3, int(args.rope_node_count))}, "
        f"rope_update_every={rope_update_every}, marker_update_every={marker_update_every}, "
        f"trail_enable={trail_enable}, rope_enable={rope_enable}, "
        f"service_timeout_ms={service_timeout_ms}"
    )
    if world_drones:
        p0 = xyz[0].copy()
        p0[2] += args.z_offset
        if world_drones_spawn:
            for spec in drone_specs:
                drone_name = str(spec["drone"])
                xy_off = np.asarray(spec["xy_off"], dtype=float)
                ok_drone = spawn_model_string(
                    args.world,
                    drone_name,
                    make_drone_model_sdf(drone_name, args.drone_mesh_uri, args.drone_scale),
                    x=float(p0[0] + xy_off[0]),
                    y=float(p0[1] + xy_off[1]),
                    z=float(p0[2] + args.drone_z_rel),
                )
                if not ok_drone:
                    ok_drone = spawn_model_string(
                        args.world,
                        drone_name,
                        make_drone_fallback_box_sdf(drone_name),
                        x=float(p0[0] + xy_off[0]),
                        y=float(p0[1] + xy_off[1]),
                        z=float(p0[2] + args.drone_z_rel),
                    )
                    if ok_drone:
                        print(f"[playback] warning: '{drone_name}' mesh unavailable; fallback box used")
                    else:
                        print(f"[playback] warning: failed to spawn '{drone_name}'")
                drone_pose_ok[drone_name] = ok_drone

        if rope_enable:
            rope_n = max(3, int(args.rope_node_count))
            for spec in drone_specs:
                drone_name = str(spec["drone"])
                rope_prefix = str(spec["rope_prefix"])
                node_names: list[str] = []
                for i in range(rope_n):
                    node_name = f"{rope_prefix}_{i:02d}"
                    ok_node = spawn_model_string(
                        args.world,
                        node_name,
                        make_rope_node_sdf(node_name, args.rope_node_radius),
                        x=float(p0[0]),
                        y=float(p0[1]),
                        z=float(p0[2] + 0.35),
                    )
                    if not ok_node:
                        print(f"[playback] warning: failed to spawn rope node '{node_name}'")
                    drone_pose_ok[node_name] = ok_node
                    node_names.append(node_name)
                rope_nodes[drone_name] = node_names

        if world_drones_spawn:
            if rope_enable:
                print("[playback] world-frame drones enabled (spawned + deformable ropes)")
            else:
                print("[playback] world-frame drones enabled (spawned, rope disabled)")
        else:
            if rope_enable:
                print("[playback] world-frame drones enabled (preloaded + deformable ropes)")
            else:
                print("[playback] world-frame drones enabled (preloaded, rope disabled)")
    start = time.monotonic()
    last_ok = True
    next_trail = 0.0
    trail_idx = 0
    last_trail_pos = None
    frame_idx = 0
    while True:
        frame_idx += 1
        elapsed = (time.monotonic() - start) * max(args.speed, 1e-6)
        pos, yaw = interpolate_pose(elapsed, t, xyz)
        pose_pos = pos.copy()
        pose_pos[2] += args.z_offset
        if attitude is not None:
            idx = min(int(np.searchsorted(t, elapsed, side="right")), len(t) - 1)
            seg = int(attitude["sample_seg"][min(idx, len(attitude["sample_seg"]) - 1)])
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
            quat = quat_from_pitch_y(theta)
        else:
            quat = quat_from_yaw(yaw)
        ok = call_set_pose(
            args.world,
            args.entity,
            pose_pos,
            quat,
            timeout_ms=service_timeout_ms,
        )
        if not ok and last_ok:
            print("[playback] warning: set_pose did not confirm success")
        last_ok = ok
        if marker_enabled and (frame_idx % marker_update_every == 0):
            marker_quat = quat_from_yaw(args.goal_marker_yaw_rate * elapsed)
            marker_ok = call_set_pose(
                args.world,
                args.goal_marker_entity,
                marker_pos,
                marker_quat,
                timeout_ms=service_timeout_ms,
            )
            if not marker_ok and marker_last_ok:
                print(
                    f"[playback] warning: marker set_pose failed for "
                    f"'{args.goal_marker_entity}'"
                )
            marker_last_ok = marker_ok
        if world_drones:
            for spec in drone_specs:
                drone_name = str(spec["drone"])
                xy_off = np.asarray(spec["xy_off"], dtype=float)
                yaw_w = float(spec["yaw_w"])
                drone_pos = np.array(
                    [
                        pose_pos[0] + float(xy_off[0]),
                        pose_pos[1] + float(xy_off[1]),
                        pose_pos[2] + args.drone_z_rel,
                    ],
                    dtype=float,
                )
                drone_ok = call_set_pose(
                    args.world,
                    drone_name,
                    drone_pos,
                    quat_from_yaw(yaw_w),
                    timeout_ms=service_timeout_ms,
                )
                if not drone_ok and drone_pose_ok.get(drone_name, True):
                    print(f"[playback] warning: drone set_pose failed for '{drone_name}'")
                drone_pose_ok[drone_name] = drone_ok

                if rope_enable:
                    anchor_local = np.asarray(spec["anchor_local"], dtype=float)
                    anchor_world = pose_pos + quat_rotate_vec(quat, anchor_local)
                    hook_world = drone_pos + np.array([0.0, 0.0, args.drone_hook_z_rel], dtype=float)
                    nodes = rope_nodes.get(drone_name, [])
                    n_nodes = len(nodes)
                    if frame_idx % rope_update_every == 0:
                        for i, node_name in enumerate(nodes):
                            u = float(i + 1) / float(n_nodes + 1)
                            node_pos = (1.0 - u) * anchor_world + u * hook_world
                            node_pos[2] -= args.rope_sag * math.sin(math.pi * u)
                            node_ok = call_set_pose(
                                args.world,
                                node_name,
                                node_pos,
                                (0.0, 0.0, 0.0, 1.0),
                                timeout_ms=service_timeout_ms,
                            )
                            if not node_ok and drone_pose_ok.get(node_name, True):
                                print(f"[playback] warning: rope node set_pose failed for '{node_name}'")
                            drone_pose_ok[node_name] = node_ok
        if trail_enable and elapsed >= next_trail:
            if last_trail_pos is not None:
                spawn_trail_segment(
                    args.world,
                    f"trail_seg_{trail_idx:04d}",
                    last_trail_pos,
                    pose_pos,
                    max(args.trail_radius * 1.8, 0.06),
                )
            last_trail_pos = pose_pos.copy()
            next_trail += max(args.trail_period, 0.1)
            trail_idx += 1
        if elapsed >= float(t[-1]):
            break
        time.sleep(max(args.dt, 0.02))

    pos, yaw = interpolate_pose(float(t[-1]), t, xyz)
    pose_pos = pos.copy()
    pose_pos[2] += args.z_offset
    if attitude is not None:
        final_seg = int(attitude["sample_seg"][-1])
        theta = sample_theta_from_export(
            pos,
            final_seg,
            attitude["hard_idx"],
            attitude["p_wp"],
            attitude["roll_wp"],
        )
        theta = args.theta_sign * theta
        if flip_enabled and (y_lo <= float(pos[1]) <= y_hi):
            theta = args.flip_theta_sign * theta
        quat = quat_from_pitch_y(theta)
    else:
        quat = quat_from_yaw(yaw)
    call_set_pose(
        args.world,
        args.entity,
        pose_pos,
        quat,
        timeout_ms=service_timeout_ms,
    )
    if marker_enabled:
        marker_quat = quat_from_yaw(args.goal_marker_yaw_rate * float(t[-1]))
        call_set_pose(
            args.world,
            args.goal_marker_entity,
            marker_pos,
            marker_quat,
            timeout_ms=service_timeout_ms,
        )
    if world_drones:
        for spec in drone_specs:
            drone_name = str(spec["drone"])
            xy_off = np.asarray(spec["xy_off"], dtype=float)
            yaw_w = float(spec["yaw_w"])
            drone_pos = np.array(
                [
                    pose_pos[0] + float(xy_off[0]),
                    pose_pos[1] + float(xy_off[1]),
                    pose_pos[2] + args.drone_z_rel,
                ],
                dtype=float,
            )
            call_set_pose(
                args.world,
                drone_name,
                drone_pos,
                quat_from_yaw(yaw_w),
                timeout_ms=service_timeout_ms,
            )
            if rope_enable:
                anchor_local = np.asarray(spec["anchor_local"], dtype=float)
                anchor_world = pose_pos + quat_rotate_vec(quat, anchor_local)
                hook_world = drone_pos + np.array([0.0, 0.0, args.drone_hook_z_rel], dtype=float)
                nodes = rope_nodes.get(drone_name, [])
                n_nodes = len(nodes)
                for i, node_name in enumerate(nodes):
                    u = float(i + 1) / float(n_nodes + 1)
                    node_pos = (1.0 - u) * anchor_world + u * hook_world
                    node_pos[2] -= args.rope_sag * math.sin(math.pi * u)
                    call_set_pose(
                        args.world,
                        node_name,
                        node_pos,
                        (0.0, 0.0, 0.0, 1.0),
                        timeout_ms=service_timeout_ms,
                    )
    print("[playback] finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
