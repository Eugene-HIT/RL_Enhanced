#!/usr/bin/env python3
"""Generate mesh assets and an SDF world for the fixed RL_enhanced scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


THIS_DIR = Path(__file__).resolve().parent
PKG_DIR = THIS_DIR.parent
MESH_DIR = PKG_DIR / "meshes"
WORLD_DIR = PKG_DIR / "worlds"

WALL_MARGIN_X = 0.65
WALL_MARGIN_Z = 0.65
PORTAL_EXTRA_DEPTH = 0.8
FRAME_COLOR = "0.65 0.67 0.72 1"
MASS_COLOR = "0.52 0.55 0.60 1"
FRAME_THICKNESS_X = 0.32
FRAME_THICKNESS_Z = 0.32


@dataclass(frozen=True)
class DoorSpec:
    name: str
    shape_type: int
    sx: float
    sz: float
    center_x: float
    center_z: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class MidObstacle:
    name: str
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


def template_polygon(shape_type: int) -> np.ndarray:
    if shape_type == 0:
        pts = np.array(
            [[-0.25, -0.5], [0.25, -0.5], [0.25, 0.5], [-0.25, 0.5]],
            dtype=float,
        )
    elif shape_type == 1:
        pts = np.array([[-0.3, -0.5], [0.3, -0.5], [-0.3, 0.5]], dtype=float)
    elif shape_type == 2:
        pts = np.array(
            [[-0.25, -0.5], [0.25, -0.4], [0.35, 0.0], [0.10, 0.5], [-0.30, 0.3]],
            dtype=float,
        )
    else:
        pts = np.array(
            [
                [-0.4, -0.6],
                [0.4, -0.6],
                [0.4, -0.3],
                [0.0, -0.3],
                [0.0, 0.3],
                [0.4, 0.3],
                [0.4, 0.6],
                [-0.4, 0.6],
            ],
            dtype=float,
        )

    return pts - pts.mean(axis=0)


def make_fixed_doors() -> list[dict]:
    specs = [
        DoorSpec("door1", 3, 1.20, 1.70, 0.90, 1.00, 0.00, 2.00),
        DoorSpec("door2", 2, 1.60, 1.50, 1.50, 1.10, 5.00, 7.00),
        DoorSpec("door3", 1, 1.20, 1.55, 1.25, 0.95, 10.00, 12.00),
        DoorSpec("door4", 0, 1.50, 1.80, 1.20, 1.05, 15.00, 17.00),
    ]
    doors = []
    for spec in specs:
        base = template_polygon(spec.shape_type)
        pts = np.column_stack([spec.sx * base[:, 0], spec.sz * base[:, 1]]) + np.array(
            [spec.center_x, spec.center_z],
            dtype=float,
        )
        poly = Polygon(pts).buffer(0.0)
        xmin, zmin, xmax, zmax = poly.bounds
        doors.append(
            {
                "name": spec.name,
                "poly": poly,
                "bbox": (xmin, zmin, xmax, zmax),
                "door_cx": 0.5 * (xmin + xmax),
                "door_cz": 0.5 * (zmin + zmax),
                "y_min": spec.y_min,
                "y_max": spec.y_max,
            }
        )
    return doors


def make_mid_obstacles(doors: list[dict]) -> list[MidObstacle]:
    obstacles = []
    # Shrink solid middle obstacles to better match current payload clearance needs.
    inflated_size = 0.92
    for i in range(len(doors) - 1):
        d1 = doors[i]
        d2 = doors[i + 1]
        cx = 0.5 * (d1["door_cx"] + d2["door_cx"])
        cz = 0.5 * (d1["door_cz"] + d2["door_cz"])
        y_mid = 0.5 * (d1["y_max"] + d2["y_min"])
        obstacles.append(
            MidObstacle(
                name=f"mid_obs_{i}",
                center_x=float(cx),
                center_y=float(y_mid),
                center_z=float(cz),
                size_x=float(inflated_size),
                size_y=1.10,
                size_z=float(inflated_size),
            )
        )
    return obstacles


def _ring_without_closure(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ring = list(coords)
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def write_wall_stl(path: Path, wall_polygon: Polygon, y_min: float, y_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_vertex(vertex: tuple[float, float, float]) -> int:
        vertices.append(vertex)
        return len(vertices)

    def add_tri(a: int, b: int, c: int) -> None:
        faces.append((a, b, c))

    top_indices: dict[tuple[float, float], int] = {}
    bottom_indices: dict[tuple[float, float], int] = {}

    def get_index(x: float, z: float, top: bool) -> int:
        key = (round(x, 8), round(z, 8))
        mapping = top_indices if top else bottom_indices
        if key not in mapping:
            mapping[key] = add_vertex((x, y_max if top else y_min, z))
        return mapping[key]

    tris = triangulate(wall_polygon)
    for tri in tris:
        probe = tri.representative_point()
        if not wall_polygon.covers(probe):
            continue
        coords = _ring_without_closure(tri.exterior.coords)
        if len(coords) != 3:
            continue
        b = [get_index(x, z, top=False) for x, z in coords]
        t = [get_index(x, z, top=True) for x, z in coords]
        add_tri(b[0], b[2], b[1])
        add_tri(t[0], t[1], t[2])

    def add_side_faces(ring_coords: Iterable[tuple[float, float]], reverse: bool) -> None:
        ring = _ring_without_closure(ring_coords)
        for i in range(len(ring)):
            x1, z1 = ring[i]
            x2, z2 = ring[(i + 1) % len(ring)]
            b1 = get_index(x1, z1, top=False)
            b2 = get_index(x2, z2, top=False)
            t1 = get_index(x1, z1, top=True)
            t2 = get_index(x2, z2, top=True)
            if reverse:
                add_tri(b1, t2, b2)
                add_tri(b1, t1, t2)
            else:
                add_tri(b1, b2, t2)
                add_tri(b1, t2, t1)

    add_side_faces(wall_polygon.exterior.coords, reverse=False)
    for hole in wall_polygon.interiors:
        add_side_faces(hole.coords, reverse=True)

    def tri_normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
        ab = np.subtract(b, a)
        ac = np.subtract(c, a)
        n = np.cross(ab, ac)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return (0.0, 0.0, 1.0)
        n = n / norm
        return (float(n[0]), float(n[1]), float(n[2]))

    with path.open("w", encoding="ascii") as f:
        f.write(f"solid {path.stem}\n")
        for a_idx, b_idx, c_idx in faces:
            a = vertices[a_idx - 1]
            b = vertices[b_idx - 1]
            c = vertices[c_idx - 1]
            nx, ny, nz = tri_normal(a, b, c)
            f.write(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
            f.write(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}\n")
            f.write(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {path.stem}\n")


def build_wall_polygon(door: dict) -> Polygon:
    xmin, zmin, xmax, zmax = door["bbox"]
    outer = Polygon(
        [
            (xmin - WALL_MARGIN_X, zmin - WALL_MARGIN_Z),
            (xmax + WALL_MARGIN_X, zmin - WALL_MARGIN_Z),
            (xmax + WALL_MARGIN_X, zmax + WALL_MARGIN_Z),
            (xmin - WALL_MARGIN_X, zmax + WALL_MARGIN_Z),
        ]
    )
    wall = outer.difference(door["poly"])
    return wall.buffer(0.0)


def mesh_uri(name: str) -> str:
    return f"../meshes/{name}.stl"


def add_box_element(
    lines: list[str],
    kind: str,
    name: str,
    size_x: float,
    size_y: float,
    size_z: float,
    pose_x: float,
    pose_y: float,
    pose_z: float,
    color: str,
) -> None:
    lines.extend(
        [
            f'        <{kind} name="{name}">',
            f'          <pose>{pose_x:.6f} {pose_y:.6f} {pose_z:.6f} 0 0 0</pose>',
            '          <geometry>',
            f'            <box><size>{size_x:.6f} {size_y:.6f} {size_z:.6f}</size></box>',
            '          </geometry>',
            *(
                [
                    '          <material>',
                    f'            <ambient>{color}</ambient>',
                    f'            <diffuse>{color}</diffuse>',
                    '          </material>',
                ]
                if kind == "visual"
                else []
            ),
            f'        </{kind}>',
        ]
    )


def build_world_sdf(doors: list[dict], obstacles: list[MidObstacle]) -> str:
    start_pose = (0.0, doors[0]["y_min"] - 2.0, 0.5)
    goal_pose = (0.0, doors[-1]["y_max"] + 2.0, 0.5)

    lines = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.10">',
        '  <world name="rl_enhanced_fixed_scene">',
        '    <gravity>0 0 -9.8</gravity>',
        '    <physics name="1ms" type="ignored">',
        '      <max_step_size>0.001</max_step_size>',
        '      <real_time_factor>1.0</real_time_factor>',
        '    </physics>',
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        '    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"/>',
        '    <light name="sun" type="directional">',
        '      <cast_shadows>true</cast_shadows>',
        '      <pose>0 0 20 0 0 0</pose>',
        '      <diffuse>0.9 0.9 0.9 1</diffuse>',
        '      <specular>0.2 0.2 0.2 1</specular>',
        '      <direction>-0.5 0.1 -0.9</direction>',
        '    </light>',
        '    <model name="ground_plane">',
        '      <static>true</static>',
        '      <link name="ground_link">',
        '        <collision name="collision">',
        '          <geometry><plane><normal>0 0 1</normal><size>80 80</size></plane></geometry>',
        '        </collision>',
        '        <visual name="visual">',
        '          <geometry><plane><normal>0 0 1</normal><size>80 80</size></plane></geometry>',
        '          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>',
        '        </visual>',
        '      </link>',
        '    </model>',
    ]

    for door in doors:
        mesh_name = f"{door['name']}_wall"
        xmin, zmin, xmax, zmax = door["bbox"]
        y_min = float(door["y_min"])
        y_max = float(door["y_max"])
        outer_xmin = xmin - WALL_MARGIN_X
        outer_xmax = xmax + WALL_MARGIN_X
        outer_zmin = zmin - WALL_MARGIN_Z
        outer_zmax = zmax + WALL_MARGIN_Z
        y_center = 0.5 * (y_min + y_max)
        portal_depth = (y_max - y_min) + 2.0 * PORTAL_EXTRA_DEPTH
        outer_width = outer_xmax - outer_xmin
        outer_height = outer_zmax - outer_zmin
        lines.extend(
            [
                f'    <model name="{mesh_name}">',
                '      <static>true</static>',
                '      <link name="link">',
                '        <collision name="collision">',
                '          <geometry>',
                '            <mesh>',
                f'              <uri>{mesh_uri(mesh_name)}</uri>',
                '            </mesh>',
                '          </geometry>',
                '        </collision>',
                '        <visual name="visual">',
                '          <geometry>',
                '            <mesh>',
                f'              <uri>{mesh_uri(mesh_name)}</uri>',
                '            </mesh>',
                '          </geometry>',
                '          <material>',
                f'            <ambient>{FRAME_COLOR}</ambient>',
                f'            <diffuse>{FRAME_COLOR}</diffuse>',
                '          </material>',
                '        </visual>',
            ]
        )

        for kind in ("collision", "visual"):
            add_box_element(
                lines,
                kind=kind,
                name=f"{kind}_left_mass",
                size_x=FRAME_THICKNESS_X,
                size_y=portal_depth,
                size_z=outer_height,
                pose_x=outer_xmin + 0.5 * FRAME_THICKNESS_X,
                pose_y=y_center,
                pose_z=0.5 * (outer_zmin + outer_zmax),
                color=MASS_COLOR,
            )
            add_box_element(
                lines,
                kind=kind,
                name=f"{kind}_right_mass",
                size_x=FRAME_THICKNESS_X,
                size_y=portal_depth,
                size_z=outer_height,
                pose_x=outer_xmax - 0.5 * FRAME_THICKNESS_X,
                pose_y=y_center,
                pose_z=0.5 * (outer_zmin + outer_zmax),
                color=MASS_COLOR,
            )
            add_box_element(
                lines,
                kind=kind,
                name=f"{kind}_top_mass",
                size_x=outer_width,
                size_y=portal_depth,
                size_z=FRAME_THICKNESS_Z,
                pose_x=0.5 * (outer_xmin + outer_xmax),
                pose_y=y_center,
                pose_z=outer_zmax - 0.5 * FRAME_THICKNESS_Z,
                color=MASS_COLOR,
            )
            add_box_element(
                lines,
                kind=kind,
                name=f"{kind}_bottom_mass",
                size_x=outer_width,
                size_y=portal_depth,
                size_z=FRAME_THICKNESS_Z,
                pose_x=0.5 * (outer_xmin + outer_xmax),
                pose_y=y_center,
                pose_z=outer_zmin + 0.5 * FRAME_THICKNESS_Z,
                color=MASS_COLOR,
            )

        lines.extend(
            [
                '      </link>',
                '    </model>',
            ]
        )

    for obs in obstacles:
        lines.extend(
            [
                f'    <model name="{obs.name}">',
                '      <static>true</static>',
                f'      <pose>{obs.center_x:.6f} {obs.center_y:.6f} {obs.center_z:.6f} 0 0 0</pose>',
                '      <link name="link">',
                '        <collision name="collision">',
                '          <geometry>',
                f'            <box><size>{obs.size_x:.6f} {obs.size_y:.6f} {obs.size_z:.6f}</size></box>',
                '          </geometry>',
                '        </collision>',
                '        <visual name="visual">',
                '          <geometry>',
                f'            <box><size>{obs.size_x:.6f} {obs.size_y:.6f} {obs.size_z:.6f}</size></box>',
                '          </geometry>',
                '          <material>',
                '            <ambient>0.78 0.34 0.30 1</ambient>',
                '            <diffuse>0.78 0.34 0.30 1</diffuse>',
                '          </material>',
                '        </visual>',
                '      </link>',
                '    </model>',
            ]
        )

    for name, pose, color in [
        ("start_marker", start_pose, "0.1 0.8 0.1 1"),
        ("goal_marker", goal_pose, "0.95 0.75 0.1 1"),
    ]:
        lines.extend(
            [
                f'    <model name="{name}">',
                '      <static>true</static>',
                f'      <pose>{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} 0 0 0</pose>',
                '      <link name="link">',
                '        <visual name="visual">',
                '          <geometry><sphere><radius>0.18</radius></sphere></geometry>',
                '          <material>',
                f'            <ambient>{color}</ambient>',
                f'            <diffuse>{color}</diffuse>',
                '          </material>',
                '        </visual>',
                '      </link>',
                '    </model>',
            ]
        )

    lines.extend(['  </world>', '</sdf>'])
    return "\n".join(lines) + "\n"


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)

    doors = make_fixed_doors()
    for door in doors:
        wall_polygon = build_wall_polygon(door)
        write_wall_stl(
            MESH_DIR / f"{door['name']}_wall.stl",
            wall_polygon=wall_polygon,
            y_min=float(door["y_min"]),
            y_max=float(door["y_max"]),
        )

    obstacles = make_mid_obstacles(doors)
    world_text = build_world_sdf(doors, obstacles)
    (WORLD_DIR / "fixed_scene_world.sdf").write_text(world_text, encoding="ascii")

    print(f"Generated {len(doors)} door meshes in {MESH_DIR}")
    print(f"Generated world file: {WORLD_DIR / 'fixed_scene_world.sdf'}")


if __name__ == "__main__":
    main()
