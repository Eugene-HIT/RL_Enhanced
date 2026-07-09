#!/usr/bin/env python3
"""Generate a static SDF overlay of payload/drone timelapse snapshots."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate timelapse snapshot overlay SDF")
    p.add_argument("--csv", required=True, help="Input CSV with columns t,x,y,z,theta")
    p.add_argument("--out", required=True, help="Output SDF model path")
    p.add_argument("--spacing", type=float, default=0.5, help="Distance spacing in meters")
    p.add_argument("--z-offset", type=float, default=1.10, help="Global z offset applied to payload")
    p.add_argument("--max-samples", type=int, default=90, help="Cap snapshot count for performance")
    p.add_argument(
        "--doorway-y",
        default="1.0,6.0,11.0,16.0",
        help="Comma-separated doorway center Y list; one snapshot just before each will be highlighted",
    )
    p.add_argument(
        "--doorway-lead",
        type=float,
        default=0.35,
        help="Highlight lead distance (m) before each doorway center along +Y direction",
    )
    p.add_argument(
        "--drone-mesh-uri",
        default="file:///home/eugene/ros2_ws/src/rl_enhanced_gz_scene_timelapse/models/payload_box/meshes/quadrotor_base.dae",
    )
    return p.parse_args()


def load_samples(csv_path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    with csv_path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(";", ",")
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                values = tuple(float(part) for part in parts)
            except ValueError:
                continue
            if len(values) not in (5, 14):
                continue
            rows.append(values)
    if len(rows) < 2:
        raise RuntimeError(f"Not enough samples in csv: {csv_path}")
    return rows


def select_by_spacing(samples: list[tuple[float, ...]], spacing: float, max_samples: int) -> list[tuple[float, ...]]:
    spacing = max(0.05, spacing)
    max_samples = max(1, max_samples)

    selected = [samples[0]]
    accum = 0.0
    prev = samples[0]

    for cur in samples[1:]:
        seg = math.sqrt((cur[1] - prev[1]) ** 2 + (cur[2] - prev[2]) ** 2 + (cur[3] - prev[3]) ** 2)
        accum += seg
        prev = cur
        if accum >= spacing:
            selected.append(cur)
            accum = 0.0
            if len(selected) >= max_samples:
                break

    if selected[-1] != samples[-1] and len(selected) < max_samples:
        selected.append(samples[-1])
    return selected


def parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for token in raw.replace(";", ",").split(","):
        s = token.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    return vals


def _sample_key(sample: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(v, 6) for v in sample)


def pick_pre_doorway_samples(
    samples: list[tuple[float, ...]],
    doorway_y_list: list[float],
    doorway_lead: float,
) -> list[tuple[float, ...]]:
    picked_highlight: list[tuple[float, ...]] = []
    if not samples or not doorway_y_list:
        return picked_highlight

    lead = max(0.0, doorway_lead)
    ys = [row[2] for row in samples]
    for door_y in doorway_y_list:
        target_y = door_y - lead
        # Prefer the nearest snapshot still before the doorway.
        best_i = None
        best_delta = float("inf")
        for i, y in enumerate(ys):
            if y <= target_y:
                delta = target_y - y
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
        if best_i is None and ys:
            # Fallback: if no "before doorway" sample exists, use nearest sample to target.
            best_i = min(range(len(ys)), key=lambda i: abs(ys[i] - target_y))
        if best_i is not None:
            picked_highlight.append(samples[best_i])
    return picked_highlight


def merge_snapshots(
    base: list[tuple[float, ...]],
    extra: list[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    merged = list(base)
    keys = {_sample_key(s) for s in base}
    for s in extra:
        k = _sample_key(s)
        if k not in keys:
            merged.append(s)
            keys.add(k)
    merged.sort(key=lambda r: r[0])
    return merged


def get_highlight_indices(
    picked: list[tuple[float, ...]],
    highlighted_samples: list[tuple[float, ...]],
) -> set[int]:
    target = {_sample_key(s) for s in highlighted_samples}
    highlighted: set[int] = set()
    for i, s in enumerate(picked):
        if _sample_key(s) in target:
            highlighted.add(i)
    return highlighted


def build_overlay_sdf(
    picked: list[tuple[float, ...]],
    z_offset: float,
    drone_mesh_uri: str,
    highlighted_indices: set[int],
) -> str:
    payload_size = (0.56, 0.26, 0.18)
    drone_scale = (0.22, 0.22, 0.22)
    drone_z_rel = 0.40
    front_offset = (0.20, 0.00)
    rear_left_offset = (-0.17, 0.13)
    rear_right_offset = (-0.17, -0.13)

    lines: list[str] = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<sdf version="1.10">')
    lines.append('  <model name="trajectory_timelapse_overlay">')
    lines.append("    <static>true</static>")
    lines.append("    <self_collide>false</self_collide>")
    lines.append("    <allow_auto_disable>true</allow_auto_disable>")

    n = max(1, len(picked) - 1)
    for i, sample in enumerate(picked):
        _t, x, y, z, theta = sample[:5]
        px, py, pz = x, y, z + z_offset
        base_alpha = 0.35 + 0.45 * (i / n)
        base_emissive = 0.10 + 0.30 * (i / n)
        is_highlight = i in highlighted_indices
        alpha = 1.0 if is_highlight else base_alpha
        emissive = 0.70 if is_highlight else base_emissive

        lines.append(f'    <link name="payload_{i:03d}">')
        lines.append(f"      <pose>{px:.6f} {py:.6f} {pz:.6f} 0.000000 {theta:.6f} 0.000000</pose>")
        lines.append('      <visual name="visual">')
        lines.append("        <geometry>")
        lines.append(
            f"          <box><size>{payload_size[0]:.3f} {payload_size[1]:.3f} {payload_size[2]:.3f}</size></box>"
        )
        lines.append("        </geometry>")
        lines.append("        <material>")
        lines.append(f"          <ambient>0.05 0.62 0.95 {alpha:.3f}</ambient>")
        lines.append(f"          <diffuse>0.05 0.62 0.95 {alpha:.3f}</diffuse>")
        lines.append(f"          <emissive>0.05 0.25 {emissive:.3f} {alpha:.3f}</emissive>")
        lines.append("        </material>")
        lines.append("        <cast_shadows>false</cast_shadows>")
        lines.append("      </visual>")
        lines.append("    </link>")

        if len(sample) >= 14:
            drone_poses = [
                (f"drone_front_{i:03d}", sample[5], sample[6], sample[7] + z_offset, 0.0),
                (f"drone_rear_left_{i:03d}", sample[8], sample[9], sample[10] + z_offset, 0.60),
                (f"drone_rear_right_{i:03d}", sample[11], sample[12], sample[13] + z_offset, -0.60),
            ]
        else:
            drone_poses = [
                (f"drone_front_{i:03d}", px + front_offset[0], py + front_offset[1], pz + drone_z_rel, 0.0),
                (f"drone_rear_left_{i:03d}", px + rear_left_offset[0], py + rear_left_offset[1], pz + drone_z_rel, 0.60),
                (
                    f"drone_rear_right_{i:03d}",
                    px + rear_right_offset[0],
                    py + rear_right_offset[1],
                    pz + drone_z_rel,
                    -0.60,
                ),
            ]

        for name, dx, dy, dz, yaw in drone_poses:
            lines.append(f'    <link name="{name}">')
            lines.append(f"      <pose>{dx:.6f} {dy:.6f} {dz:.6f} 0.000000 0.000000 {yaw:.6f}</pose>")
            lines.append('      <visual name="visual">')
            lines.append("        <geometry>")
            lines.append("          <mesh>")
            lines.append(f"            <uri>{drone_mesh_uri}</uri>")
            lines.append(
                f"            <scale>{drone_scale[0]:.6f} {drone_scale[1]:.6f} {drone_scale[2]:.6f}</scale>"
            )
            lines.append("          </mesh>")
            lines.append("        </geometry>")
            lines.append("        <material>")
            if is_highlight:
                lines.append("          <ambient>0.05 0.62 0.95 1.000</ambient>")
                lines.append("          <diffuse>0.05 0.62 0.95 1.000</diffuse>")
                lines.append("          <emissive>0.05 0.30 0.90 1.000</emissive>")
            else:
                lines.append(f"          <ambient>0.92 0.92 0.92 {alpha:.3f}</ambient>")
                lines.append(f"          <diffuse>0.92 0.92 0.92 {alpha:.3f}</diffuse>")
                lines.append(f"          <emissive>0.20 0.20 0.20 {alpha:.3f}</emissive>")
            lines.append("        </material>")
            lines.append("        <cast_shadows>false</cast_shadows>")
            lines.append("      </visual>")
            lines.append("    </link>")

    lines.append("  </model>")
    lines.append("</sdf>")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    samples = load_samples(csv_path)
    picked_base = select_by_spacing(samples, args.spacing, args.max_samples)
    doorway_y = parse_float_list(args.doorway_y)
    highlighted_samples = pick_pre_doorway_samples(samples, doorway_y, args.doorway_lead)
    picked = merge_snapshots(picked_base, highlighted_samples)
    highlighted = get_highlight_indices(picked, highlighted_samples)
    sdf_text = build_overlay_sdf(picked, args.z_offset, args.drone_mesh_uri, highlighted)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sdf_text, encoding="ascii")
    highlight_str = ",".join(str(i) for i in sorted(highlighted)) if highlighted else "none"
    print(
        f"[timelapse_overlay] input_samples={len(samples)}, picked={len(picked)}, "
        f"spacing={args.spacing:.3f}m, highlighted_indices={highlight_str}, out={out_path}"
    )


if __name__ == "__main__":
    main()
