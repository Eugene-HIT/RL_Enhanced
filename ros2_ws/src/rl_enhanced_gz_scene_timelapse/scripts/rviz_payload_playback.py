#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from scipy.io import loadmat
from visualization_msgs.msg import Marker, MarkerArray


def interpolate_position(elapsed: float, t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if elapsed <= float(t[0]):
        return xyz[0]
    elif elapsed >= float(t[-1]):
        return xyz[-1]

    idx = int(np.searchsorted(t, elapsed, side="right"))
    i0 = max(0, idx - 1)
    i1 = min(len(t) - 1, idx)
    alpha = (elapsed - float(t[i0])) / max(float(t[i1] - t[i0]), 1e-9)
    return (1.0 - alpha) * xyz[i0] + alpha * xyz[i1]


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def sample_theta_from_export(pos: np.ndarray, seg_id: int, hard_idx: np.ndarray, p_wp: np.ndarray, roll_wp: np.ndarray) -> float:
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


class PayloadPlaybackNode(Node):
    def __init__(self, xyz_path: str, t_path: str, speed: float, export_path: str | None) -> None:
        super().__init__("rviz_payload_playback")
        self.xyz = np.load(xyz_path).astype(float)
        self.t = np.load(t_path).astype(float)
        self.speed = max(speed, 1e-6)
        self.start = time.monotonic()
        self.last_line_stamp = 0.0
        self.trail_points: list[Point] = []
        self.sample_seg: np.ndarray | None = None
        self.hard_idx: np.ndarray | None = None
        self.p_wp: np.ndarray | None = None
        self.roll_wp: np.ndarray | None = None
        self.doors = []
        self.forbidden = []
        if export_path:
            mat = loadmat(export_path, squeeze_me=True, struct_as_record=False)
            traj = mat["traj"]
            key = mat["keyframes"]
            self.sample_seg = np.asarray(getattr(traj, "sample_seg"), dtype=int).reshape(-1)
            self.hard_idx = np.asarray(getattr(traj, "hard_idx"), dtype=int).reshape(-1)
            self.p_wp = np.asarray(getattr(key, "P_wp"), dtype=float)
            self.roll_wp = np.asarray(getattr(key, "roll_wp"), dtype=float).reshape(-1)
            self.doors = list(np.atleast_1d(mat["doors"]))
            self.forbidden = list(np.atleast_1d(mat["forbidden"]))
        self.publisher = self.create_publisher(MarkerArray, "/payload_playback_markers", 10)
        self.timer = self.create_timer(0.05, self.on_timer)
        self.get_logger().info(
            f"Loaded trajectory: samples={self.xyz.shape[0]}, duration={float(self.t[-1]):.3f}s, speed={self.speed:.2f}x"
        )
        if self.sample_seg is not None:
            self.get_logger().info(f"Loaded attitude reference from {export_path}")

    def on_timer(self) -> None:
        elapsed = (time.monotonic() - self.start) * self.speed
        pos = interpolate_position(elapsed, self.t, self.xyz)
        idx = min(int(np.searchsorted(self.t, elapsed, side="right")), len(self.t) - 1)
        if self.sample_seg is not None and self.hard_idx is not None and self.p_wp is not None and self.roll_wp is not None:
            pitch_y = sample_theta_from_export(pos, int(self.sample_seg[idx]), self.hard_idx, self.p_wp, self.roll_wp)
        else:
            i0 = max(0, idx - 1)
            i1 = min(len(self.t) - 1, idx)
            delta_xz = self.xyz[i1, [0, 2]] - self.xyz[i0, [0, 2]]
            pitch_y = math.atan2(float(delta_xz[1]), float(delta_xz[0])) if np.linalg.norm(delta_xz) > 1e-6 else 0.0

        if elapsed - self.last_line_stamp >= 0.25 or not self.trail_points:
            p = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            self.trail_points.append(p)
            self.last_line_stamp = elapsed

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        markers.markers.extend(self.build_scene_markers(now))

        payload = Marker()
        payload.header.frame_id = "world"
        payload.header.stamp = now
        payload.ns = "payload"
        payload.id = 1
        payload.type = Marker.CUBE
        payload.action = Marker.ADD
        payload.pose.position.x = float(pos[0])
        payload.pose.position.y = float(pos[1])
        payload.pose.position.z = float(pos[2])
        payload.pose.orientation.y = math.sin(0.5 * pitch_y)
        payload.pose.orientation.w = math.cos(0.5 * pitch_y)
        payload.scale.x = 0.95
        payload.scale.y = 0.34
        payload.scale.z = 0.22
        payload.color.r = 0.10
        payload.color.g = 0.78
        payload.color.b = 0.92
        payload.color.a = 1.0
        payload.lifetime = Duration(sec=0, nanosec=0)
        markers.markers.append(payload)

        line = Marker()
        line.header.frame_id = "world"
        line.header.stamp = now
        line.ns = "payload"
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.r = 0.98
        line.color.g = 0.82
        line.color.b = 0.18
        line.color.a = 1.0
        line.points = list(self.trail_points)
        line.lifetime = Duration(sec=0, nanosec=0)
        markers.markers.append(line)

        self.publisher.publish(markers)

        if elapsed >= float(self.t[-1]):
            self.get_logger().info("Playback finished")
            self.timer.cancel()

    def build_scene_markers(self, now) -> list[Marker]:
        markers: list[Marker] = []

        start = Marker()
        start.header.frame_id = "world"
        start.header.stamp = now
        start.ns = "scene"
        start.id = 100
        start.type = Marker.CYLINDER
        start.action = Marker.ADD
        start.pose.position.x = float(self.xyz[0, 0])
        start.pose.position.y = float(self.xyz[0, 1])
        start.pose.position.z = 0.03
        start.pose.orientation.w = 1.0
        start.scale.x = 0.8
        start.scale.y = 0.8
        start.scale.z = 0.06
        start.color.r = 0.15
        start.color.g = 0.85
        start.color.b = 0.35
        start.color.a = 0.95
        markers.append(start)

        goal = Marker()
        goal.header.frame_id = "world"
        goal.header.stamp = now
        goal.ns = "scene"
        goal.id = 101
        goal.type = Marker.CYLINDER
        goal.action = Marker.ADD
        goal.pose.position.x = float(self.xyz[-1, 0])
        goal.pose.position.y = float(self.xyz[-1, 1])
        goal.pose.position.z = 0.03
        goal.pose.orientation.w = 1.0
        goal.scale.x = 0.8
        goal.scale.y = 0.8
        goal.scale.z = 0.06
        goal.color.r = 0.95
        goal.color.g = 0.78
        goal.color.b = 0.12
        goal.color.a = 0.95
        markers.append(goal)

        for i, door in enumerate(self.doors):
            poly = np.asarray(getattr(door, "poly_xz"), dtype=float)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            y0 = float(getattr(door, "y_min"))
            y1 = float(getattr(door, "y_max"))
            cx = float(np.mean(poly[:, 0]))
            cz = float(np.mean(poly[:, 1]))
            door_marker = Marker()
            door_marker.header.frame_id = "world"
            door_marker.header.stamp = now
            door_marker.ns = "scene"
            door_marker.id = 110 + i
            door_marker.type = Marker.LINE_LIST
            door_marker.action = Marker.ADD
            door_marker.pose.orientation.w = 1.0
            door_marker.scale.x = 0.05
            door_marker.color.r = 0.72
            door_marker.color.g = 0.82
            door_marker.color.b = 0.90
            door_marker.color.a = 0.95
            n = poly.shape[0]
            for k in range(n):
                a = poly[k]
                b = poly[(k + 1) % n]
                door_marker.points.append(Point(x=float(a[0]), y=y0, z=float(a[1])))
                door_marker.points.append(Point(x=float(b[0]), y=y0, z=float(b[1])))
                door_marker.points.append(Point(x=float(a[0]), y=y1, z=float(a[1])))
                door_marker.points.append(Point(x=float(b[0]), y=y1, z=float(b[1])))
                door_marker.points.append(Point(x=float(a[0]), y=y0, z=float(a[1])))
                door_marker.points.append(Point(x=float(a[0]), y=y1, z=float(a[1])))
            markers.append(door_marker)

            label = Marker()
            label.header.frame_id = "world"
            label.header.stamp = now
            label.ns = "scene_label"
            label.id = 210 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cx
            label.pose.position.y = 0.5 * (y0 + y1)
            label.pose.position.z = cz + 0.55
            label.pose.orientation.w = 1.0
            label.scale.z = 0.34
            label.color.r = 0.86
            label.color.g = 0.92
            label.color.b = 1.0
            label.color.a = 0.95
            label.text = str(getattr(door, "name", f"door{i+1}"))
            markers.append(label)

        for i, obs in enumerate(self.forbidden):
            poly = np.asarray(getattr(obs, "poly_xz"), dtype=float)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            xmin = float(np.min(poly[:, 0]))
            xmax = float(np.max(poly[:, 0]))
            zmin = float(np.min(poly[:, 1]))
            zmax = float(np.max(poly[:, 1]))
            y0 = float(getattr(obs, "y_min"))
            y1 = float(getattr(obs, "y_max"))
            cube = Marker()
            cube.header.frame_id = "world"
            cube.header.stamp = now
            cube.ns = "scene"
            cube.id = 120 + i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = 0.5 * (xmin + xmax)
            cube.pose.position.y = 0.5 * (y0 + y1)
            cube.pose.position.z = 0.5 * (zmin + zmax)
            cube.pose.orientation.w = 1.0
            cube.scale.x = xmax - xmin
            cube.scale.y = y1 - y0
            cube.scale.z = zmax - zmin
            cube.color.r = 0.38
            cube.color.g = 0.28
            cube.color.b = 0.22
            cube.color.a = 0.72
            markers.append(cube)

            label = Marker()
            label.header.frame_id = "world"
            label.header.stamp = now
            label.ns = "scene_label"
            label.id = 220 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = 0.5 * (xmin + xmax)
            label.pose.position.y = 0.5 * (y0 + y1)
            label.pose.position.z = zmax + 0.55
            label.pose.orientation.w = 1.0
            label.scale.z = 0.34
            label.color.r = 0.96
            label.color.g = 0.88
            label.color.b = 0.70
            label.color.a = 0.95
            label.text = str(getattr(obs, "tag", f"mid_obs_{i}"))
            markers.append(label)

        return markers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--time", dest="time_path", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--export", default="/home/eugene/RL_enhanced/corridor_export.mat")
    args = parser.parse_args()

    rclpy.init()
    node = PayloadPlaybackNode(args.xyz, args.time_path, args.speed, args.export)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
