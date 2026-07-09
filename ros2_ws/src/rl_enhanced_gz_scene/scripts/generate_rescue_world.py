#!/usr/bin/env python3
"""Generate a stable rescue-ruins Gazebo world with simple collision geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate


THIS_DIR = Path(__file__).resolve().parent
PKG_DIR = THIS_DIR.parent
WORLD_DIR = PKG_DIR / "worlds"
MESH_DIR = PKG_DIR / "meshes"
FUEL_MODEL_DIR = Path("/home/eugene/.gz/fuel/fuel.gazebosim.org/openrobotics/models")
# Raise custom-generated scene geometry (walls/doors/obstacles/markers/lights)
# so trajectories are less likely to appear below ground.
SCENE_Z_OFFSET = 1.10


@dataclass(frozen=True)
class BoxModel:
    name: str
    size_x: float
    size_y: float
    size_z: float
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    color: str = "0.58 0.56 0.52 1"
    ambient: str | None = None
    diffuse: str | None = None
    specular: str = "0.08 0.08 0.08 1"
    emissive: str = "0 0 0 1"
    static: bool = True
    collide: bool = True
    mesh_uri: str | None = None
    mesh_scale_x: float | None = None
    mesh_scale_y: float | None = None
    mesh_scale_z: float | None = None
    collision_mesh: bool = False
    transparency: float = 0.0
    use_material: bool = True


@dataclass(frozen=True)
class CylinderModel:
    name: str
    radius: float
    length: float
    x: float
    y: float
    z: float
    color: str
    collide: bool = False


@dataclass(frozen=True)
class IncludeModel:
    name: str
    uri: str
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    static: bool = True


def add_box_model(lines: list[str], box: BoxModel) -> None:
    ambient = box.ambient if box.ambient is not None else box.color
    diffuse = box.diffuse if box.diffuse is not None else box.color
    if box.mesh_uri is not None:
        sx = box.mesh_scale_x if box.mesh_scale_x is not None else box.size_x
        sy = box.mesh_scale_y if box.mesh_scale_y is not None else box.size_y
        sz = box.mesh_scale_z if box.mesh_scale_z is not None else box.size_z
        visual_geometry = [
            '          <geometry>',
            '            <mesh>',
            f'              <uri>{box.mesh_uri}</uri>',
            f'              <scale>{sx:.6f} {sy:.6f} {sz:.6f}</scale>',
            '            </mesh>',
            '          </geometry>',
        ]
    else:
        visual_geometry = [
            '          <geometry>',
            f'            <box><size>{box.size_x:.6f} {box.size_y:.6f} {box.size_z:.6f}</size></box>',
            '          </geometry>',
        ]
    lines.extend(
        [
            f'    <model name="{box.name}">',
            f'      <static>{"true" if box.static else "false"}</static>',
            f'      <pose>{box.x:.6f} {box.y:.6f} {box.z + SCENE_Z_OFFSET:.6f} {box.roll:.6f} {box.pitch:.6f} {box.yaw:.6f}</pose>',
            '      <link name="link">',
        ]
    )
    if box.collide:
        lines.append('        <collision name="collision">')
        if box.mesh_uri is not None and box.collision_mesh:
            lines.extend(
                [
                    '          <geometry>',
                    '            <mesh>',
                    f'              <uri>{box.mesh_uri}</uri>',
                    f'              <scale>{sx:.6f} {sy:.6f} {sz:.6f}</scale>',
                    '            </mesh>',
                    '          </geometry>',
                ]
            )
        else:
            lines.extend(
                [
                    '          <geometry>',
                    f'            <box><size>{box.size_x:.6f} {box.size_y:.6f} {box.size_z:.6f}</size></box>',
                    '          </geometry>',
                ]
            )
        lines.append('        </collision>')
    lines.append('        <visual name="visual">')
    lines.extend(visual_geometry)
    if box.use_material:
        lines.extend(
            [
                '          <material>',
                f'            <ambient>{ambient}</ambient>',
                f'            <diffuse>{diffuse}</diffuse>',
                f'            <specular>{box.specular}</specular>',
                f'            <emissive>{box.emissive}</emissive>',
                '          </material>',
            ]
        )
    lines.extend(
        [
            '          <cast_shadows>true</cast_shadows>',
            f'          <transparency>{box.transparency:.3f}</transparency>',
            '        </visual>',
            '      </link>',
            '    </model>',
        ]
    )


def mesh_ref(name: str) -> str:
    return f"../meshes/{name}"


def fuel_mesh_uri(model_name: str, mesh_name: str) -> str | None:
    model_dir = FUEL_MODEL_DIR / model_name
    if not model_dir.exists():
        return None
    versions = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not versions:
        return None
    mesh_path = versions[-1] / "meshes" / mesh_name
    if not mesh_path.exists():
        return None
    return mesh_path.as_uri()


def write_stl(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    verts = np.asarray(vertices, dtype=float)
    lines = ["solid mesh"]
    for a, b, c in faces:
        p0 = verts[a - 1]
        p1 = verts[b - 1]
        p2 = verts[c - 1]
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            n = n / norm
        lines.append(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}")
        lines.append(f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}")
        lines.append(f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
        # Duplicate the triangle with reversed winding for visual robustness.
        # Gazebo / Ogre may back-face cull some procedurally generated facets.
        lines.append(f"  facet normal {-n[0]:.6f} {-n[1]:.6f} {-n[2]:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}")
        lines.append(f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}")
        lines.append(f"      vertex {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid mesh")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def prism_mesh(points_xz: list[tuple[float, float]], depth: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    half = 0.5 * depth
    n = len(points_xz)
    front = [(x, -half, z) for x, z in points_xz]
    back = [(x, half, z) for x, z in points_xz]
    vertices = front + back
    faces: list[tuple[int, int, int]] = []
    for i in range(1, n - 1):
        faces.append((1, i + 1, i + 2))
        faces.append((n + 1, n + i + 2, n + i + 1))
    for i in range(n):
        j = (i + 1) % n
        a, b = i + 1, j + 1
        c, d = n + j + 1, n + i + 1
        faces.append((a, b, c))
        faces.append((a, c, d))
    return vertices, faces


def rock_mesh(seed: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    corners = np.array(
        [
            [-0.50, -0.50, -0.40],
            [0.48, -0.42, -0.32],
            [0.42, 0.50, -0.48],
            [-0.46, 0.36, -0.36],
            [-0.34, -0.38, 0.48],
            [0.52, -0.32, 0.34],
            [0.30, 0.46, 0.52],
            [-0.52, 0.34, 0.40],
        ],
        dtype=float,
    )
    corners += rng.normal(scale=0.08, size=corners.shape)
    vertices = [tuple(v) for v in corners]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    return vertices, faces


def ensure_ruins_meshes() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh_defs = {
        "rubble_a.stl": rock_mesh(3),
        "rubble_b.stl": rock_mesh(7),
        "rubble_c.stl": rock_mesh(11),
        "slab_a.stl": prism_mesh([(-0.50, -0.45), (0.45, -0.50), (0.52, 0.10), (0.12, 0.50), (-0.48, 0.38)], 0.32),
        "slab_b.stl": prism_mesh([(-0.52, -0.32), (0.30, -0.48), (0.54, -0.08), (0.34, 0.46), (-0.44, 0.52)], 0.28),
        "wall_chunk_a.stl": prism_mesh([(-0.50, -0.50), (0.46, -0.42), (0.50, 0.34), (0.08, 0.52), (-0.50, 0.46)], 0.46),
        "wall_chunk_b.stl": prism_mesh([(-0.46, -0.54), (0.52, -0.48), (0.44, 0.18), (0.18, 0.54), (-0.52, 0.48)], 0.42),
    }
    for name, (vertices, faces) in mesh_defs.items():
        write_stl(MESH_DIR / name, vertices, faces)


def _ring_without_closure(coords) -> list[tuple[float, float]]:
    ring = list(coords)
    if ring and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def write_extruded_polygon_stl(
    path: Path,
    poly: Polygon,
    y_min: float,
    y_max: float,
    center_x: float,
    center_z: float,
) -> None:
    poly = poly.buffer(0.0)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    top_indices: dict[tuple[float, float], int] = {}
    bottom_indices: dict[tuple[float, float], int] = {}
    y_mid = 0.5 * (y_min + y_max)

    def add_vertex(v: tuple[float, float, float]) -> int:
        vertices.append(v)
        return len(vertices)

    def get_index(x: float, z: float, top: bool) -> int:
        key = (round(x, 8), round(z, 8))
        mapping = top_indices if top else bottom_indices
        if key not in mapping:
            yy = y_max if top else y_min
            mapping[key] = add_vertex((x - center_x, yy - y_mid, z - center_z))
        return mapping[key]

    for tri in triangulate(poly):
        probe = tri.representative_point()
        if not poly.covers(probe):
            continue
        coords = _ring_without_closure(tri.exterior.coords)
        if len(coords) != 3:
            continue
        b = [get_index(x, z, top=False) for x, z in coords]
        t = [get_index(x, z, top=True) for x, z in coords]
        faces.append((b[0], b[2], b[1]))
        faces.append((t[0], t[1], t[2]))

    def add_side_faces(ring_coords, reverse: bool) -> None:
        ring = _ring_without_closure(ring_coords)
        for i in range(len(ring)):
            x1, z1 = ring[i]
            x2, z2 = ring[(i + 1) % len(ring)]
            b1 = get_index(x1, z1, top=False)
            b2 = get_index(x2, z2, top=False)
            t1 = get_index(x1, z1, top=True)
            t2 = get_index(x2, z2, top=True)
            if reverse:
                faces.append((b1, t2, b2))
                faces.append((b1, t1, t2))
            else:
                faces.append((b1, b2, t2))
                faces.append((b1, t2, t1))

    add_side_faces(poly.exterior.coords, reverse=False)
    for hole in poly.interiors:
        add_side_faces(hole.coords, reverse=True)

    write_stl(path, vertices, faces)


def build_door_wall_polygon(door: dict, margin_x: float = 0.62, margin_z: float = 0.62) -> Polygon:
    xmin, zmin, xmax, zmax = door["bbox"]
    outer = Polygon(
        [
            (xmin - margin_x, zmin - margin_z),
            (xmax + margin_x, zmin - margin_z),
            (xmax + margin_x, zmax + margin_z),
            (xmin - margin_x, zmax + margin_z),
        ]
    )
    return outer.difference(door["poly"]).buffer(0.0)


def build_door_shell_polygon(door: dict, margin: float = 0.40) -> Polygon:
    poly = door["poly"].buffer(0.0)
    xmin, zmin, xmax, zmax = door["bbox"]
    st = int(door.get("shape_type", 0))
    side = -1.0 if st in (3, 1) else 1.0
    # Build an irregular ruin outline from bbox expansion (not a door-shaped offset).
    left_ext = margin + (0.10 if st in (1, 3) else 0.04)
    right_ext = margin + (0.10 if st in (0, 2) else 0.04)
    bottom_ext = margin * (0.95 if st in (0, 1) else 0.85)
    top_ext = margin * (1.20 if st in (2, 3) else 1.05)
    cx = 0.5 * (xmin + xmax)
    outer = Polygon(
        [
            (xmin - left_ext, zmin - bottom_ext * 0.85),
            (cx - 0.20, zmin - bottom_ext),
            (xmax + right_ext * 0.78, zmin - bottom_ext * 0.92),
            (xmax + right_ext, zmin + 0.10),
            (xmax + right_ext * 0.86, zmax + top_ext * 0.60),
            (cx + 0.14 * side, zmax + top_ext),
            (xmin - left_ext * 0.82, zmax + top_ext * 0.72),
            (xmin - left_ext, zmin + 0.10),
        ]
    ).buffer(0.06, join_style=1).buffer(0.0)

    # Add asymmetric bulges so silhouette looks like collapsed masonry.
    bulges = [
        Point(xmin - 0.12, zmax - 0.12).buffer(0.14, resolution=10),
        Point(xmax + 0.14, zmax - 0.30).buffer(0.12, resolution=10),
        Point(cx + side * 0.22, zmin + 0.20).buffer(0.11, resolution=10),
    ]
    for b in bulges:
        outer = outer.union(b)
    shell = outer.difference(poly).buffer(0.0)
    if shell.is_empty:
        return build_door_wall_polygon(door, margin_x=margin, margin_z=margin).buffer(0.0)
    return shell


def _to_polygons(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon) and (not g.is_empty)]
    return []


def _difference_many(poly: Polygon, cutters: list[Polygon]) -> Polygon:
    out = poly
    for cutter in cutters:
        out = out.difference(cutter)
    return out.buffer(0.0)


def _split_poly_with_gap(poly: Polygon, axis: str, split_value: float, gap: float) -> list[Polygon]:
    if axis == "z":
        lo = Polygon([(-100.0, -100.0), (100.0, -100.0), (100.0, split_value - 0.5 * gap), (-100.0, split_value - 0.5 * gap)])
        hi = Polygon([(-100.0, split_value + 0.5 * gap), (100.0, split_value + 0.5 * gap), (100.0, 100.0), (-100.0, 100.0)])
    else:
        lo = Polygon([(-100.0, -100.0), (split_value - 0.5 * gap, -100.0), (split_value - 0.5 * gap, 100.0), (-100.0, 100.0)])
        hi = Polygon([(split_value + 0.5 * gap, -100.0), (100.0, -100.0), (100.0, 100.0), (split_value + 0.5 * gap, 100.0)])
    out: list[Polygon] = []
    for clip in (lo, hi):
        out.extend([p for p in _to_polygons(poly.intersection(clip).buffer(0.0)) if p.area > 0.010])
    return out


def door_segment_polygons(door: dict, margin_x: float = 0.62, margin_z: float = 0.62) -> dict[str, list[Polygon]]:
    del margin_x, margin_z
    xmin, zmin, xmax, zmax = door["bbox"]
    cx = 0.5 * (xmin + xmax)
    wall_poly = build_door_shell_polygon(door, margin=0.40)
    outer_xmin, outer_zmin, outer_xmax, outer_zmax = wall_poly.bounds

    # Split doorway wall into a few independent chunks so it doesn't look machined.
    left_clip = Polygon(
        [
            (outer_xmin, outer_zmin),
            (cx - 0.02, outer_zmin),
            (cx - 0.02, outer_zmax),
            (outer_xmin, outer_zmax),
        ]
    )
    right_clip = Polygon(
        [
            (cx + 0.02, outer_zmin),
            (outer_xmax, outer_zmin),
            (outer_xmax, outer_zmax),
            (cx + 0.02, outer_zmax),
        ]
    )
    top_clip = Polygon(
        [
            (xmin - 0.30, zmax - 0.05),
            (xmax + 0.30, zmax - 0.05),
            (xmax + 0.30, outer_zmax),
            (xmin - 0.30, outer_zmax),
        ]
    )

    st = int(door.get("shape_type", 0))
    left_base = wall_poly.intersection(left_clip).buffer(0.0)
    right_base = wall_poly.intersection(right_clip).buffer(0.0)
    top_base = wall_poly.intersection(top_clip).buffer(0.0)

    # Carve true voids to create broken edges and local missing chunks.
    left_cutters = [
        Point(outer_xmin + 0.12, zmax - (0.10 if st in (3, 1) else 0.22)).buffer(0.13, resolution=14),
        Point(outer_xmin + 0.28, zmin + 0.28).buffer(0.10, resolution=12),
        Polygon(
            [
                (outer_xmin + 0.15, zmin + 0.24),
                (outer_xmin + 0.23, zmin + 0.24),
                (outer_xmin + 0.23, zmax - 0.16),
                (outer_xmin + 0.15, zmax - 0.16),
            ]
        ),
    ]
    right_cutters = [
        Point(outer_xmax - 0.12, zmax - (0.24 if st in (3, 1) else 0.10)).buffer(0.14, resolution=14),
        Point(outer_xmax - 0.30, zmin + 0.26).buffer(0.09, resolution=12),
        Polygon(
            [
                (outer_xmax - 0.23, zmin + 0.24),
                (outer_xmax - 0.15, zmin + 0.24),
                (outer_xmax - 0.15, zmax - 0.16),
                (outer_xmax - 0.23, zmax - 0.16),
            ]
        ),
    ]
    top_cutters = [
        Polygon(
            [
                (cx - 0.22, zmax + 0.18),
                (cx + 0.14, zmax + 0.16),
                (cx + 0.16, zmax + 0.42),
                (cx - 0.24, zmax + 0.46),
            ]
        ),
        Point(cx + (0.18 if st in (3, 1) else -0.18), zmax + 0.36).buffer(0.10, resolution=10),
        Polygon(
            [
                (cx - 0.09, zmax + 0.06),
                (cx + 0.09, zmax + 0.06),
                (cx + 0.09, outer_zmax - 0.04),
                (cx - 0.09, outer_zmax - 0.04),
            ]
        ),
    ]

    left_poly = _difference_many(left_base, left_cutters)
    right_poly = _difference_many(right_base, right_cutters)
    top_poly = _difference_many(top_base, top_cutters)

    segments: dict[str, list[Polygon]] = {}
    for name, geom, fallback in (
        ("core_left", left_poly, left_base),
        ("core_right", right_poly, right_base),
        ("core_top", top_poly, top_base),
    ):
        polys = [p for p in _to_polygons(geom) if p.area > 0.015]
        if not polys:
            polys = [p for p in _to_polygons(fallback) if p.area > 0.015]
        if not polys:
            polys = [wall_poly]
        # Force visible fragmentation when a segment remains monolithic.
        if len(polys) == 1:
            p = polys[0]
            if name in ("core_left", "core_right"):
                midz = 0.5 * (p.bounds[1] + p.bounds[3])
                split = _split_poly_with_gap(p, axis="z", split_value=midz, gap=0.08)
                if len(split) >= 2:
                    polys = split
            elif name == "core_top":
                midx = 0.5 * (p.bounds[0] + p.bounds[2])
                split = _split_poly_with_gap(p, axis="x", split_value=midx, gap=0.10)
                if len(split) >= 2:
                    polys = split
        # Keep all meaningful components so doorway looks visibly broken, not monolithic.
        segments[name] = [p.buffer(0.0) for p in polys if p.area > 0.010]
    return segments


def ensure_door_meshes(doors: list[dict]) -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for door in doors:
        y_min = float(door["y_min"]) - 0.02
        y_max = float(door["y_max"]) + 0.02
        cx = 0.5 * (door["bbox"][0] + door["bbox"][2])
        cz = 0.5 * (door["bbox"][1] + door["bbox"][3])
        for suffix, polys in door_segment_polygons(door).items():
            for i, poly in enumerate(polys):
                write_extruded_polygon_stl(MESH_DIR / f"{door['name']}_{suffix}_{i}.stl", poly, y_min, y_max, cx, cz)


def obstacle_visual_polygons(obs: dict) -> list[Polygon]:
    core = obs["poly"].buffer(0.0)
    xmin, zmin, xmax, zmax = obs["bbox"]
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    sx = xmax - xmin
    sz = zmax - zmin

    # Build a chunkier irregular mass around the logical obstacle core.
    mass = core.buffer(0.18, join_style=2).buffer(0.04, join_style=1)
    bulges = [
        Point(cx - 0.36 * sx, cz + 0.26 * sz).buffer(0.16, resolution=12),
        Point(cx + 0.34 * sx, cz - 0.22 * sz).buffer(0.15, resolution=12),
        Point(cx + 0.08 * sx, cz + 0.40 * sz).buffer(0.13, resolution=12),
    ]
    for b in bulges:
        mass = mass.union(b)

    # Carve chunks out to get broken / collapsed silhouette.
    chips = [
        Point(cx - 0.30 * sx, cz - 0.32 * sz).buffer(0.14, resolution=10),
        Point(cx + 0.28 * sx, cz + 0.30 * sz).buffer(0.12, resolution=10),
        Polygon(
            [
                (cx - 0.10 * sx, cz - 0.55 * sz),
                (cx + 0.18 * sx, cz - 0.48 * sz),
                (cx + 0.12 * sx, cz - 0.20 * sz),
                (cx - 0.14 * sx, cz - 0.28 * sz),
            ]
        ),
    ]
    mass = _difference_many(mass, chips)

    clip_a = Polygon(
        [
            (cx - 2.0, cz - 2.0),
            (cx - 0.06, cz - 2.0),
            (cx + 0.16, cz + 2.0),
            (cx - 2.0, cz + 2.0),
        ]
    )
    clip_b = Polygon(
        [
            (cx - 0.28, cz - 2.0),
            (cx + 0.26, cz - 2.0),
            (cx + 0.48, cz + 2.0),
            (cx - 0.08, cz + 2.0),
        ]
    )
    clip_c = Polygon(
        [
            (cx + 0.10, cz - 2.0),
            (cx + 2.0, cz - 2.0),
            (cx + 2.0, cz + 2.0),
            (cx - 0.08, cz + 2.0),
        ]
    )

    polys: list[Polygon] = []
    for clip in (clip_a, clip_b, clip_c):
        seg = mass.intersection(clip).buffer(0.0)
        polys.extend([p for p in _to_polygons(seg) if p.area > 0.02])

    if not polys:
        polys = [p for p in _to_polygons(mass) if p.area > 0.02]
    if not polys:
        polys = [core]
    return polys


def ensure_obstacle_meshes(obstacles: list[dict]) -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for obs in obstacles:
        xmin, zmin, xmax, zmax = obs["bbox"]
        cx = 0.5 * (xmin + xmax)
        cz = 0.5 * (zmin + zmax)
        y_min = float(obs["y_min"])
        y_max = float(obs["y_max"])
        write_extruded_polygon_stl(
            MESH_DIR / f"{obs['name']}_core.stl",
            obs["poly"],
            y_min,
            y_max,
            cx,
            cz,
        )
        for j, poly in enumerate(obstacle_visual_polygons(obs)):
            write_extruded_polygon_stl(
                MESH_DIR / f"{obs['name']}_ruin_{j}.stl",
                poly,
                y_min + 0.02,
                y_max - 0.02,
                cx,
                cz,
            )


def add_cylinder_model(lines: list[str], cyl: CylinderModel) -> None:
    lines.extend(
        [
            f'    <model name="{cyl.name}">',
            '      <static>true</static>',
            f'      <pose>{cyl.x:.6f} {cyl.y:.6f} {cyl.z + SCENE_Z_OFFSET:.6f} 1.570796 0 0</pose>',
            '      <link name="link">',
        ]
    )
    if cyl.collide:
        lines.extend(
            [
                '        <collision name="collision">',
                '          <geometry>',
                f'            <cylinder><radius>{cyl.radius:.6f}</radius><length>{cyl.length:.6f}</length></cylinder>',
                '          </geometry>',
                '        </collision>',
            ]
        )
    lines.extend(
        [
            '        <visual name="visual">',
            '          <geometry>',
            f'            <cylinder><radius>{cyl.radius:.6f}</radius><length>{cyl.length:.6f}</length></cylinder>',
            '          </geometry>',
            '          <material>',
            f'            <ambient>{cyl.color}</ambient>',
            f'            <diffuse>{cyl.color}</diffuse>',
            f'            <emissive>{cyl.color}</emissive>',
            '          </material>',
            '          <transparency>0.18</transparency>',
            '        </visual>',
            '      </link>',
            '    </model>',
        ]
    )


def add_rotating_goal_cross_model(
    lines: list[str],
    name: str,
    x: float,
    y: float,
    z: float,
) -> None:
    lines.extend(
        [
            f'    <model name="{name}">',
            '      <static>false</static>',
            f'      <pose>{x:.6f} {y:.6f} {z + SCENE_Z_OFFSET:.6f} 0 0 0</pose>',
            '      <link name="link">',
            '        <gravity>false</gravity>',
            '        <inertial>',
            '          <mass>0.010</mass>',
            '          <inertia>',
            '            <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>',
            '            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>',
            '          </inertia>',
            '        </inertial>',
            '        <visual name="base_ring">',
            '          <pose>0 0 -0.210000 0 0 0</pose>',
            '          <geometry><cylinder><radius>0.300000</radius><length>0.090000</length></cylinder></geometry>',
            '          <material>',
            '            <ambient>0.12 0.78 0.30 1</ambient>',
            '            <diffuse>0.18 0.92 0.36 1</diffuse>',
            '            <specular>0.05 0.10 0.05 1</specular>',
            '            <emissive>0.10 0.45 0.15 1</emissive>',
            '          </material>',
            '          <cast_shadows>true</cast_shadows>',
            '        </visual>',
            '      </link>',
            '    </model>',
        ]
    )


def add_injured_person_model(
    lines: list[str],
    name: str,
    x: float,
    y: float,
    z: float,
    yaw: float = 0.25,
) -> None:
    lines.extend(
        [
            f'    <model name="{name}">',
            '      <static>true</static>',
            f'      <pose>{x:.6f} {y:.6f} {z + SCENE_Z_OFFSET:.6f} 0 0 {yaw:.6f}</pose>',
            '      <link name="link">',
            # Torso
            '        <visual name="torso">',
            '          <pose>0.000 0.000 0.160 0 0 0</pose>',
            '          <geometry><box><size>0.56 0.24 0.14</size></box></geometry>',
            '          <material>',
            '            <ambient>0.82 0.42 0.14 1</ambient>',
            '            <diffuse>0.92 0.50 0.18 1</diffuse>',
            '            <specular>0.03 0.03 0.03 1</specular>',
            '          </material>',
            '        </visual>',
            # Head
            '        <visual name="head">',
            '          <pose>0.330 0.000 0.200 0 0 0</pose>',
            '          <geometry><sphere><radius>0.085</radius></sphere></geometry>',
            '          <material>',
            '            <ambient>0.68 0.52 0.42 1</ambient>',
            '            <diffuse>0.84 0.66 0.54 1</diffuse>',
            '            <specular>0.02 0.02 0.02 1</specular>',
            '          </material>',
            '        </visual>',
            # Left leg
            '        <visual name="leg_l">',
            '          <pose>-0.220 0.070 0.115 0 0 0</pose>',
            '          <geometry><box><size>0.34 0.09 0.10</size></box></geometry>',
            '          <material>',
            '            <ambient>0.12 0.16 0.30 1</ambient>',
            '            <diffuse>0.20 0.26 0.44 1</diffuse>',
            '            <specular>0.02 0.02 0.03 1</specular>',
            '          </material>',
            '        </visual>',
            # Right leg
            '        <visual name="leg_r">',
            '          <pose>-0.220 -0.070 0.115 0 0 0</pose>',
            '          <geometry><box><size>0.34 0.09 0.10</size></box></geometry>',
            '          <material>',
            '            <ambient>0.12 0.16 0.30 1</ambient>',
            '            <diffuse>0.20 0.26 0.44 1</diffuse>',
            '            <specular>0.02 0.02 0.03 1</specular>',
            '          </material>',
            '        </visual>',
            # Left arm
            '        <visual name="arm_l">',
            '          <pose>0.090 0.190 0.160 0 0 0.250</pose>',
            '          <geometry><box><size>0.28 0.07 0.07</size></box></geometry>',
            '          <material>',
            '            <ambient>0.66 0.50 0.40 1</ambient>',
            '            <diffuse>0.82 0.64 0.52 1</diffuse>',
            '            <specular>0.02 0.02 0.02 1</specular>',
            '          </material>',
            '        </visual>',
            # Right arm
            '        <visual name="arm_r">',
            '          <pose>0.090 -0.190 0.150 0 0 -0.380</pose>',
            '          <geometry><box><size>0.30 0.07 0.07</size></box></geometry>',
            '          <material>',
            '            <ambient>0.66 0.50 0.40 1</ambient>',
            '            <diffuse>0.82 0.64 0.52 1</diffuse>',
            '            <specular>0.02 0.02 0.02 1</specular>',
            '          </material>',
            '        </visual>',
            '      </link>',
            '    </model>',
        ]
    )


def add_include_model(lines: list[str], inc: IncludeModel) -> None:
    lines.extend(
        [
            "    <include>",
            f"      <name>{inc.name}</name>",
            f"      <uri>{inc.uri}</uri>",
            f"      <pose>{inc.x:.6f} {inc.y:.6f} {inc.z:.6f} {inc.roll:.6f} {inc.pitch:.6f} {inc.yaw:.6f}</pose>",
            f'      <static>{"true" if inc.static else "false"}</static>',
            "    </include>",
        ]
    )


def add_point_light(
    lines: list[str],
    name: str,
    x: float,
    y: float,
    z: float,
    diffuse: str = "1.0 0.42 0.08 1",
    specular: str = "0.15 0.08 0.02 1",
    range_v: float = 6.0,
    constant: float = 0.25,
    linear: float = 0.08,
    quadratic: float = 0.02,
) -> None:
    lines.extend(
        [
            f'    <light name="{name}" type="point">',
            f'      <pose>{x:.6f} {y:.6f} {z + SCENE_Z_OFFSET:.6f} 0 0 0</pose>',
            f'      <diffuse>{diffuse}</diffuse>',
            f'      <specular>{specular}</specular>',
            '      <attenuation>',
            f'        <range>{range_v:.3f}</range>',
            f'        <constant>{constant:.3f}</constant>',
            f'        <linear>{linear:.3f}</linear>',
            f'        <quadratic>{quadratic:.3f}</quadratic>',
            '      </attenuation>',
            '      <cast_shadows>true</cast_shadows>',
            '    </light>',
        ]
    )


def add_doorway_lights(lines: list[str], doors: list[dict]) -> None:
    for door in doors:
        xmin, zmin, xmax, zmax = door["bbox"]
        cx = 0.5 * (xmin + xmax)
        y0 = float(door["y_min"])
        y1 = float(door["y_max"])
        y_mid = 0.5 * (y0 + y1)
        depth = max(0.2, y1 - y0)
        h = max(0.2, zmax - zmin)
        z_inner = zmin + 0.42 * h
        # Illuminate the traversable interior volume directly (not the top beam).
        for i, yy in enumerate((y0 + 0.22 * depth, y_mid, y1 - 0.22 * depth)):
            add_point_light(
                lines,
                f"{door['name']}_inner_light_{i}",
                cx,
                yy,
                z_inner,
                diffuse="0.96 0.72 0.22 1",
                specular="0.12 0.09 0.04 1",
                range_v=2.8,
                constant=0.24,
                linear=0.26,
                quadratic=0.08,
            )
        # A low floor-near fill light helps reveal passable boundary at the base.
        add_point_light(
            lines,
            f"{door['name']}_inner_floor_fill",
            cx,
            y_mid,
            zmin + 0.12,
            diffuse="0.90 0.68 0.20 1",
            specular="0.10 0.08 0.03 1",
            range_v=2.2,
            constant=0.30,
            linear=0.32,
            quadratic=0.11,
        )


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
        ("door1", 3, 1.20, 1.70, 0.90, 1.00, 0.00, 2.00),
        ("door2", 2, 1.60, 1.50, 1.50, 1.10, 5.00, 7.00),
        ("door3", 1, 1.20, 1.55, 1.25, 0.95, 10.00, 12.00),
        ("door4", 0, 1.50, 1.80, 1.20, 1.05, 15.00, 17.00),
    ]
    doors = []
    for name, shape_type, sx, sz, cx, cz, y_min, y_max in specs:
        base = template_polygon(shape_type)
        pts = np.column_stack([sx * base[:, 0], sz * base[:, 1]]) + np.array([cx, cz], dtype=float)
        poly = Polygon(pts).buffer(0.0)
        xmin, zmin, xmax, zmax = poly.bounds
        doors.append(
            {
                "name": name,
                "shape_type": shape_type,
                "poly": poly,
                "bbox": (float(xmin), float(zmin), float(xmax), float(zmax)),
                "y_min": float(y_min),
                "y_max": float(y_max),
            }
        )
    return doors


def door_damage_regions(door: dict, outer_xmin: float, outer_xmax: float, outer_zmin: float, outer_zmax: float) -> list[tuple[float, float, float]]:
    mid_x = 0.5 * (outer_xmin + outer_xmax)
    top_z = outer_zmax
    low_z = outer_zmin
    shape_type = int(door.get("shape_type", 0))
    if shape_type == 3:
        return [
            (outer_xmin + 0.22, top_z - 0.18, 0.32),
            (outer_xmax - 0.12, top_z - 0.28, 0.26),
            (outer_xmax - 0.18, low_z + 0.22, 0.22),
        ]
    if shape_type == 2:
        return [
            (outer_xmin + 0.18, top_z - 0.20, 0.28),
            (outer_xmax - 0.16, top_z - 0.38, 0.34),
            (mid_x + 0.36, low_z + 0.26, 0.20),
        ]
    if shape_type == 1:
        return [
            (outer_xmin + 0.20, top_z - 0.18, 0.30),
            (outer_xmax - 0.20, top_z - 0.24, 0.22),
            (outer_xmin + 0.22, low_z + 0.24, 0.18),
        ]
    return [
        (outer_xmin + 0.16, top_z - 0.16, 0.26),
        (outer_xmax - 0.20, top_z - 0.22, 0.30),
        (mid_x - 0.42, low_z + 0.20, 0.16),
    ]


def block_is_damaged(door: dict, center_x: float, center_z: float, outer_xmin: float, outer_xmax: float, outer_zmin: float, outer_zmax: float) -> bool:
    for cx, cz, radius in door_damage_regions(door, outer_xmin, outer_xmax, outer_zmin, outer_zmax):
        if (center_x - cx) ** 2 + (center_z - cz) ** 2 <= radius ** 2:
            return True
    return False


def door_block_boxes(
    door: dict,
    margin_x: float = 0.70,
    margin_z: float = 0.70,
    cell: float = 0.18,
) -> list[BoxModel]:
    xmin, zmin, xmax, zmax = door["bbox"]
    outer_xmin = xmin - margin_x
    outer_xmax = xmax + margin_x
    outer_zmin = zmin - margin_z
    outer_zmax = zmax + margin_z
    y_center = 0.5 * (door["y_min"] + door["y_max"])
    depth = (door["y_max"] - door["y_min"]) + 0.04
    opening = door["poly"]
    boxes: list[BoxModel] = []

    z = outer_zmin + 0.5 * cell
    row_idx = 0
    eps = 0.015
    while z < outer_zmax:
        x = outer_xmin + 0.5 * cell
        run_start = None
        while x < outer_xmax:
            blocked = not opening.covers(Point(x, z))
            if blocked and run_start is None:
                run_start = x
            if (not blocked or x + cell >= outer_xmax) and run_start is not None:
                end_x = x if not blocked else x + cell
                width = max(end_x - run_start, cell)
                center_x = 0.5 * (run_start + end_x)
                if not block_is_damaged(door, center_x, z, outer_xmin, outer_xmax, outer_zmin, outer_zmax):
                    size_z = cell + eps
                    z_offset = 0.0
                    # Add slight irregularity to surviving blocks so the doorway looks chipped, not machined.
                    if (row_idx + int(abs(center_x) * 10)) % 5 == 0:
                        size_z = max(0.11, size_z - 0.035)
                        z_offset = 0.012
                    boxes.append(
                        BoxModel(
                            name=f"{door['name']}_blk_{row_idx}_{len(boxes)}",
                            size_x=width + eps,
                            size_y=depth,
                            size_z=size_z,
                            x=center_x,
                            y=y_center,
                            z=z + z_offset,
                            color="0.60 0.59 0.56 1",
                        )
                    )
                run_start = None
            x += cell
        z += cell
        row_idx += 1

    return boxes


def centered_doors() -> list[dict]:
    doors = make_fixed_doors()
    x_shift = 0.0
    for door in doors:
        door["poly"] = Polygon([(x + x_shift, z) for x, z in door["poly"].exterior.coords[:-1]]).buffer(0.0)
        xmin, zmin, xmax, zmax = door["poly"].bounds
        door["bbox"] = (float(xmin), float(zmin), float(xmax), float(zmax))
    return doors


def centered_mid_obstacles() -> list[dict]:
    doors = centered_doors()
    out = []
    # Keep fixed-scene layout, but shrink solid obstacles to reduce unnecessary collisions.
    size = 0.72
    inflate = 0.08
    half = 0.5 * size
    for i in range(len(doors) - 1):
        d1 = doors[i]
        d2 = doors[i + 1]
        cx = 0.5 * (0.5 * (d1["bbox"][0] + d1["bbox"][2]) + 0.5 * (d2["bbox"][0] + d2["bbox"][2]))
        cz = 0.5 * (0.5 * (d1["bbox"][1] + d1["bbox"][3]) + 0.5 * (d2["bbox"][1] + d2["bbox"][3]))
        y_mid = 0.5 * (float(d1["y_max"]) + float(d2["y_min"]))
        poly = Polygon(
            [
                (cx - half, cz - half),
                (cx + half, cz - half),
                (cx + half, cz + half),
                (cx - half, cz + half),
            ]
        ).buffer(inflate)
        xmin, zmin, xmax, zmax = poly.bounds
        out.append(
            {
                "name": f"mid_obs_{i}",
                "poly": poly,
                "bbox": (float(xmin), float(zmin), float(xmax), float(zmax)),
                "y_min": float(y_mid - 0.55),
                "y_max": float(y_mid + 0.55),
            }
        )
    return out


def route_center_x(doors: list[dict]) -> float:
    if not doors:
        return 0.0
    mids = [0.5 * (d["bbox"][0] + d["bbox"][2]) for d in doors]
    return float(np.mean(np.asarray(mids, dtype=float)))


def trajectory_center_x(default: float) -> float:
    p = Path("/home/eugene/RL_enhanced/qp3d_sample_xyz.npy")
    if not p.exists():
        return default
    try:
        xyz = np.load(p).astype(float)
    except Exception:
        return default
    if xyz.ndim != 2 or xyz.shape[0] < 1 or xyz.shape[1] < 1:
        return default
    return float(np.mean(xyz[:, 0]))


def trajectory_center_range(default: tuple[float, float]) -> tuple[float, float]:
    p = Path("/home/eugene/RL_enhanced/qp3d_sample_xyz.npy")
    if not p.exists():
        return default
    try:
        xyz = np.load(p).astype(float)
    except Exception:
        return default
    if xyz.ndim != 2 or xyz.shape[0] < 1 or xyz.shape[1] < 1:
        return default
    return float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))


def scene_layout_x(doors: list[dict], obstacles: list[dict]) -> tuple[float, float]:
    xmins: list[float] = []
    xmaxs: list[float] = []
    for door in doors:
        for seg_polys in door_segment_polygons(door).values():
            for seg in seg_polys:
                xmin, _, xmax, _ = seg.bounds
                xmins.append(float(xmin))
                xmaxs.append(float(xmax))
    for obs in obstacles:
        xmin, _, xmax, _ = obs["bbox"]
        xmins.append(float(xmin))
        xmaxs.append(float(xmax))
    if not xmins:
        return 0.0, 1.6
    geom_center = 0.5 * (min(xmins) + max(xmaxs))
    traj_min, traj_max = trajectory_center_range((geom_center - 0.8, geom_center + 0.8))
    traj_center = 0.5 * (traj_min + traj_max)
    center = 0.75 * traj_center + 0.25 * geom_center
    # Widen side clearance to reduce visual clipping when adding textured ruins.
    half = max(center - min(xmins), max(xmaxs) - center) + 0.60
    return float(center), float(max(2.40, min(half, 3.40)))


def start_goal_markers() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.0, -2.0, 0.5], dtype=float), np.array([0.0, 19.0, 0.5], dtype=float)


def doorway_ruin_boxes(door: dict) -> list[BoxModel]:
    xmin, zmin, xmax, zmax = door["bbox"]
    y0 = float(door["y_min"])
    y1 = float(door["y_max"])
    yc = 0.5 * (y0 + y1)
    cz = 0.5 * (zmin + zmax)
    width = xmax - xmin
    height = zmax - zmin
    depth = (y1 - y0) + 0.04
    cx = 0.5 * (xmin + xmax)

    concrete = "0.58 0.55 0.50 1"
    dark = "0.44 0.36 0.30 1"
    boxes: list[BoxModel] = []
    st = int(door["shape_type"])
    margin_x = 0.40
    margin_z = 0.40
    outer_w = width + 2.0 * margin_x
    outer_h = height + 2.0 * margin_z
    pieces = door_segment_polygons(door, margin_x=margin_x, margin_z=margin_z)
    for seg, polys in pieces.items():
        keep_polys = polys
        # Leave a deliberate top void for ruined look while keeping doorway clearance untouched.
        if seg == "core_top" and len(polys) > 1:
            drop_idx = min(range(len(polys)), key=lambda j: abs(polys[j].centroid.x - cx))
            keep_polys = [p for j, p in enumerate(polys) if j != drop_idx]
            if not keep_polys:
                keep_polys = [polys[0]]
        for i, poly in enumerate(keep_polys):
            pxmin, pzmin, pxmax, pzmax = poly.bounds
            pw = max(0.08, pxmax - pxmin)
            ph = max(0.08, pzmax - pzmin)
            boxes.append(
                BoxModel(
                    f"{door['name']}_{seg}_{i}",
                    pw,
                    depth,
                    ph,
                    cx,
                    yc,
                    cz,
                    color=concrete,
                    ambient="0.24 0.22 0.20 1",
                    diffuse="0.46 0.42 0.37 1",
                    specular="0.012 0.012 0.012 1",
                    mesh_uri=mesh_ref(f"{door['name']}_{seg}_{i}.stl"),
                    mesh_scale_x=1.0,
                    mesh_scale_y=1.0,
                    mesh_scale_z=1.0,
                    collision_mesh=True,
                    transparency=0.0,
                )
            )

    # Keep only a few broken exterior fragments so door topology remains obvious.
    side_sign = -1.0 if st in (3, 1) else 1.0
    chunk_mesh = "wall_chunk_b.stl" if side_sign < 0 else "wall_chunk_a.stl"
    top_mesh = "slab_b.stl" if side_sign < 0 else "slab_a.stl"
    boxes.append(
        BoxModel(
            f"{door['name']}_broken_side",
            0.34,
            depth * 0.85,
            max(0.95, height * 0.76),
            cx + side_sign * (0.5 * outer_w - 0.06),
            yc + side_sign * 0.08,
            zmin + 0.46 * max(0.95, height * 0.76),
            color=dark,
            ambient="0.20 0.16 0.12 1",
            diffuse="0.36 0.28 0.22 1",
            specular="0.010 0.008 0.006 1",
            collide=False,
            mesh_uri=mesh_ref(chunk_mesh),
            mesh_scale_x=0.36,
            mesh_scale_y=depth * 0.85,
            mesh_scale_z=max(0.95, height * 0.76),
        )
    )
    boxes.append(
        BoxModel(
            f"{door['name']}_broken_top",
            max(0.40, width * 0.30),
            depth * 0.62,
            0.18,
            cx + side_sign * 0.30,
            yc + side_sign * 0.12,
            zmax + 0.20,
            yaw=-0.18 * side_sign,
            color=dark,
            ambient="0.14 0.14 0.13 1",
            diffuse="0.28 0.27 0.25 1",
            specular="0.008 0.008 0.008 1",
            collide=False,
            mesh_uri=mesh_ref(top_mesh),
            mesh_scale_x=max(0.40, width * 0.30),
            mesh_scale_y=depth * 0.62,
            mesh_scale_z=0.22,
        )
    )
    boxes.append(
        BoxModel(
            f"{door['name']}_floor_frag",
            0.24,
            depth * 0.28,
            0.14,
            cx - side_sign * 0.18,
            yc + 0.10,
            zmin + 0.10,
            yaw=0.16 * side_sign,
            color=dark,
            ambient="0.20 0.17 0.13 1",
            diffuse="0.34 0.28 0.22 1",
            specular="0.010 0.008 0.006 1",
            collide=False,
            mesh_uri=mesh_ref("rubble_a.stl"),
            mesh_scale_x=0.20,
            mesh_scale_y=depth * 0.26,
            mesh_scale_z=0.13,
        )
    )
    boxes.append(
        BoxModel(
            f"{door['name']}_ledge_frag",
            0.30,
            depth * 0.36,
            0.12,
            cx + side_sign * 0.08,
            yc - side_sign * 0.10,
            zmax + 0.08,
            yaw=0.22 * side_sign,
            pitch=0.12,
            color=dark,
            ambient="0.12 0.12 0.12 1",
            diffuse="0.24 0.24 0.23 1",
            specular="0.008 0.008 0.008 1",
            collide=False,
            mesh_uri=mesh_ref("rubble_b.stl"),
            mesh_scale_x=0.22,
            mesh_scale_y=depth * 0.30,
            mesh_scale_z=0.11,
        )
    )
    boxes.append(
        BoxModel(
            f"{door['name']}_base_frag",
            0.26,
            depth * 0.30,
            0.10,
            cx - side_sign * 0.26,
            yc - 0.06,
            zmin + 0.06,
            yaw=-0.18 * side_sign,
            color=dark,
            ambient="0.18 0.14 0.10 1",
            diffuse="0.30 0.23 0.18 1",
            specular="0.010 0.008 0.006 1",
            collide=False,
            mesh_uri=mesh_ref("rubble_c.stl"),
            mesh_scale_x=0.20,
            mesh_scale_y=depth * 0.24,
            mesh_scale_z=0.09,
        )
    )

    return boxes


def corridor_shell_boxes(center_x: float, half_width: float) -> list[BoxModel]:
    concrete = "0.50 0.47 0.42 1"
    dark = "0.33 0.31 0.29 1"
    dust_1 = {"ambient": "0.21 0.19 0.16 1", "diffuse": "0.40 0.35 0.30 1", "specular": "0.012 0.012 0.012 1"}
    dust_2 = {"ambient": "0.18 0.16 0.13 1", "diffuse": "0.34 0.30 0.26 1", "specular": "0.010 0.010 0.010 1"}
    rust_1 = {"ambient": "0.20 0.14 0.10 1", "diffuse": "0.34 0.24 0.17 1", "specular": "0.008 0.007 0.006 1"}
    char_1 = {"ambient": "0.09 0.08 0.07 1", "diffuse": "0.15 0.14 0.13 1", "specular": "0.004 0.004 0.004 1"}
    wall_t = 0.32
    left = center_x - half_width - 0.5 * wall_t
    right = center_x + half_width + 0.5 * wall_t
    floor_w = 2.0 * (half_width + 0.16)
    return [
        BoxModel("rear_block", floor_w, 0.42, 2.5, center_x, -2.1, 1.25, color=concrete, ambient=dust_1["ambient"], diffuse=dust_1["diffuse"], specular=dust_1["specular"]),
        BoxModel("left_wall_a", wall_t, 3.9, 2.3, left - 0.03, 0.2, 1.16, yaw=0.03, color=concrete, ambient=dust_1["ambient"], diffuse=dust_1["diffuse"], specular=dust_1["specular"]),
        BoxModel("left_wall_b", wall_t, 4.2, 2.3, left + 0.06, 5.0, 1.16, yaw=-0.05, color=concrete, ambient=rust_1["ambient"], diffuse=rust_1["diffuse"], specular=rust_1["specular"]),
        BoxModel("left_wall_c", wall_t, 3.8, 2.3, left - 0.05, 9.8, 1.16, yaw=0.04, color=concrete, ambient=dust_2["ambient"], diffuse=dust_2["diffuse"], specular=dust_2["specular"]),
        BoxModel("left_wall_d", wall_t, 3.6, 2.3, left + 0.08, 14.4, 1.16, yaw=-0.03, color=concrete, ambient=char_1["ambient"], diffuse=char_1["diffuse"], specular=char_1["specular"]),
        BoxModel("left_wall_e", wall_t, 3.7, 2.3, left - 0.02, 18.2, 1.16, yaw=0.02, color=concrete, ambient=dust_1["ambient"], diffuse=dust_1["diffuse"], specular=dust_1["specular"]),
        BoxModel("right_wall_a", wall_t, 3.8, 2.3, right + 0.03, 0.3, 1.16, yaw=-0.04, color=concrete, ambient=rust_1["ambient"], diffuse=rust_1["diffuse"], specular=rust_1["specular"]),
        BoxModel("right_wall_b", wall_t, 4.1, 2.3, right - 0.06, 5.1, 1.16, yaw=0.05, color=concrete, ambient=dust_2["ambient"], diffuse=dust_2["diffuse"], specular=dust_2["specular"]),
        BoxModel("right_wall_c", wall_t, 3.9, 2.3, right + 0.05, 10.0, 1.16, yaw=-0.03, color=concrete, ambient=char_1["ambient"], diffuse=char_1["diffuse"], specular=char_1["specular"]),
        BoxModel("right_wall_d", wall_t, 3.7, 2.3, right - 0.08, 14.9, 1.16, yaw=0.04, color=concrete, ambient=dust_1["ambient"], diffuse=dust_1["diffuse"], specular=dust_1["specular"]),
        BoxModel("right_wall_e", wall_t, 3.5, 2.3, right + 0.03, 18.1, 1.16, yaw=-0.02, color=concrete, ambient=rust_1["ambient"], diffuse=rust_1["diffuse"], specular=rust_1["specular"]),
        BoxModel("ceiling_a", 1.35, 2.2, 0.18, center_x - 0.82, 2.4, 2.10, pitch=0.10, yaw=0.14, color=dark, ambient=dust_2["ambient"], diffuse=dust_2["diffuse"], specular=dust_2["specular"]),
        BoxModel("ceiling_b", 1.24, 2.4, 0.18, center_x + 0.86, 8.1, 2.06, pitch=-0.08, yaw=-0.12, color=dark, ambient=char_1["ambient"], diffuse=char_1["diffuse"], specular=char_1["specular"]),
        BoxModel("ceiling_c", 1.30, 2.1, 0.18, center_x - 0.78, 13.9, 2.12, pitch=0.12, yaw=0.10, color=dark, ambient=rust_1["ambient"], diffuse=rust_1["diffuse"], specular=rust_1["specular"]),
    ]


def obstacle_ruin_boxes(obstacles: list[dict]) -> list[BoxModel]:
    ash = "0.20 0.20 0.19 1"
    concrete = "0.52 0.50 0.46 1"
    ruin_palette = [
        {
            "ambient": "0.24 0.22 0.20 1",
            "diffuse": "0.46 0.42 0.37 1",
            "specular": "0.018 0.018 0.018 1",
        },
        {
            "ambient": "0.22 0.16 0.11 1",
            "diffuse": "0.40 0.29 0.21 1",
            "specular": "0.011 0.009 0.007 1",
        },
        {
            "ambient": "0.10 0.10 0.10 1",
            "diffuse": "0.18 0.18 0.17 1",
            "specular": "0.006 0.006 0.006 1",
        },
    ]
    boxes: list[BoxModel] = []
    for i, obs in enumerate(obstacles):
        xmin, zmin, xmax, zmax = obs["bbox"]
        yc = 0.5 * (obs["y_min"] + obs["y_max"])
        cx = 0.5 * (xmin + xmax)
        cz = 0.5 * (zmin + zmax)
        sx = xmax - xmin
        sz = zmax - zmin
        sy = obs["y_max"] - obs["y_min"]
        boxes.append(
            BoxModel(
                f"{obs['name']}_core",
                sx,
                sy,
                sz,
                cx,
                yc,
                cz,
                color=ash,
                mesh_uri=mesh_ref(f"{obs['name']}_core.stl"),
                mesh_scale_x=1.0,
                mesh_scale_y=1.0,
                mesh_scale_z=1.0,
                collision_mesh=True,
                transparency=0.995,
                specular="0.005 0.005 0.005 1",
            )
        )
        for j, poly in enumerate(obstacle_visual_polygons(obs)):
            pxmin, pzmin, pxmax, pzmax = poly.bounds
            pw = max(0.08, pxmax - pxmin)
            ph = max(0.08, pzmax - pzmin)
            mat = ruin_palette[(i + j) % len(ruin_palette)]
            boxes.append(
                BoxModel(
                    f"{obs['name']}_ruin_{j}",
                    pw,
                    sy,
                    ph,
                    cx,
                    yc,
                    cz,
                    pitch=(0.07 if j % 2 == 0 else -0.05) * (1.0 if i != 1 else -1.0),
                    yaw=(0.10 if j == 0 else (-0.08 if j == 1 else 0.06)),
                    color=concrete if j != 1 else ash,
                    ambient=mat["ambient"],
                    diffuse=mat["diffuse"],
                    specular=mat["specular"],
                    collide=False,
                    mesh_uri=mesh_ref(f"{obs['name']}_ruin_{j}.stl"),
                    mesh_scale_x=1.0,
                    mesh_scale_y=1.0,
                    mesh_scale_z=1.0,
                    transparency=0.0,
                )
            )
        boxes.append(
            BoxModel(
                f"{obs['name']}_crown_frag",
                0.30 + 0.06 * i,
                sy * 0.55,
                0.14,
                cx + (0.22 if i % 2 == 0 else -0.24),
                yc,
                zmax + 0.28,
                pitch=0.24 if i % 2 == 0 else -0.22,
                yaw=0.11 if i != 1 else -0.10,
                color=ash,
                ambient="0.17 0.13 0.10 1",
                diffuse="0.40 0.29 0.21 1",
                specular="0.011 0.009 0.007 1",
                collide=False,
                mesh_uri=mesh_ref("slab_b.stl"),
                mesh_scale_x=0.24 + 0.05 * i,
                mesh_scale_y=sy * 0.48,
                mesh_scale_z=0.16,
            )
        )
    return boxes


def side_rubble_boxes(center_x: float, half_width: float) -> list[BoxModel]:
    ash = "0.18 0.17 0.16 1"
    concrete = "0.43 0.39 0.33 1"
    left_x = center_x - (half_width - 0.16)
    right_x = center_x + (half_width - 0.16)
    return [
        BoxModel("rubble_left_0", 0.80, 0.62, 0.36, left_x, 5.8, 0.19, yaw=-0.12, color=concrete, ambient="0.20 0.18 0.15 1", diffuse="0.37 0.33 0.27 1", specular="0.010 0.010 0.010 1", collide=False, mesh_uri=mesh_ref("wall_chunk_a.stl")),
        BoxModel("rubble_left_1", 0.64, 0.50, 0.30, left_x - 0.03, 14.9, 0.16, yaw=0.14, color=ash, ambient="0.16 0.12 0.09 1", diffuse="0.31 0.23 0.17 1", specular="0.008 0.007 0.006 1", collide=False, mesh_uri=mesh_ref("rubble_b.stl")),
        BoxModel("rubble_right_0", 0.74, 0.58, 0.34, right_x, 7.1, 0.18, yaw=0.14, color=concrete, ambient="0.08 0.08 0.08 1", diffuse="0.16 0.16 0.15 1", specular="0.004 0.004 0.004 1", collide=False, mesh_uri=mesh_ref("wall_chunk_b.stl")),
        BoxModel("rubble_right_1", 0.60, 0.48, 0.28, right_x + 0.02, 16.1, 0.15, yaw=-0.15, color=ash, ambient="0.16 0.12 0.09 1", diffuse="0.31 0.23 0.17 1", specular="0.008 0.007 0.006 1", collide=False, mesh_uri=mesh_ref("rubble_a.stl")),
    ]


def decal_boxes(center_x: float, half_width: float) -> list[BoxModel]:
    left_face_x = center_x - half_width + 0.010
    right_face_x = center_x + half_width - 0.010
    return [
        # Left wall graffiti layers
        BoxModel("graffiti_l_base", 0.040, 1.20, 0.36, left_face_x, 4.8, 1.15, color="0.70 0.20 0.16 1", ambient="0.52 0.12 0.10 1", diffuse="0.70 0.20 0.16 1", specular="0.002 0.002 0.002 1", emissive="0.03 0.01 0.01 1", collide=False, transparency=0.00),
        BoxModel("graffiti_l_tag", 0.040, 0.80, 0.20, left_face_x + 0.001, 4.98, 1.25, yaw=0.12, color="0.96 0.74 0.16 1", ambient="0.62 0.44 0.08 1", diffuse="0.96 0.74 0.16 1", specular="0.002 0.002 0.002 1", emissive="0.06 0.04 0.01 1", collide=False, transparency=0.00),
        BoxModel("graffiti_l_drip", 0.038, 0.30, 0.26, left_face_x + 0.002, 5.17, 0.98, yaw=-0.10, color="0.10 0.10 0.10 1", ambient="0.08 0.08 0.08 1", diffuse="0.18 0.18 0.18 1", specular="0.001 0.001 0.001 1", collide=False, transparency=0.02),
        # Right wall stencil + spray
        BoxModel("graffiti_r_stencil", 0.040, 1.00, 0.32, right_face_x, 11.1, 1.10, color="0.18 0.50 0.72 1", ambient="0.10 0.30 0.42 1", diffuse="0.18 0.50 0.72 1", specular="0.002 0.002 0.002 1", emissive="0.01 0.03 0.05 1", collide=False, transparency=0.00),
        BoxModel("graffiti_r_tag", 0.040, 0.70, 0.18, right_face_x - 0.001, 11.0, 1.24, yaw=-0.15, color="0.94 0.54 0.18 1", ambient="0.58 0.28 0.08 1", diffuse="0.94 0.54 0.18 1", specular="0.002 0.002 0.002 1", emissive="0.05 0.03 0.01 1", collide=False, transparency=0.00),
        # Caution stripe pattern pieces near obstacle area
        BoxModel("stripe_l_0", 0.036, 0.30, 0.09, left_face_x + 0.001, 8.6, 1.30, yaw=0.30, color="0.94 0.78 0.14 1", ambient="0.54 0.43 0.08 1", diffuse="0.94 0.78 0.14 1", specular="0.003 0.003 0.003 1", collide=False, transparency=0.00),
        BoxModel("stripe_l_1", 0.036, 0.30, 0.09, left_face_x + 0.001, 8.87, 1.22, yaw=0.30, color="0.10 0.10 0.10 1", ambient="0.07 0.07 0.07 1", diffuse="0.16 0.16 0.16 1", specular="0.002 0.002 0.002 1", collide=False, transparency=0.00),
        BoxModel("stripe_l_2", 0.036, 0.30, 0.09, left_face_x + 0.001, 9.14, 1.14, yaw=0.30, color="0.94 0.78 0.14 1", ambient="0.54 0.43 0.08 1", diffuse="0.94 0.78 0.14 1", specular="0.003 0.003 0.003 1", collide=False, transparency=0.00),
        # Floor soot / oil stains
        BoxModel("stain_floor_0", 0.74, 0.52, 0.010, center_x + 0.34, 6.3, 0.095, yaw=0.22, color="0.06 0.06 0.06 1", ambient="0.04 0.04 0.04 1", diffuse="0.11 0.11 0.11 1", specular="0.001 0.001 0.001 1", collide=False, transparency=0.15),
        BoxModel("stain_floor_1", 0.60, 0.46, 0.010, center_x - 0.46, 12.4, 0.095, yaw=-0.30, color="0.10 0.07 0.04 1", ambient="0.05 0.04 0.02 1", diffuse="0.14 0.10 0.06 1", specular="0.001 0.001 0.001 1", collide=False, transparency=0.20),
    ]


def environment_includes(center_x: float, half_width: float) -> list[IncludeModel]:
    del center_x, half_width
    return [
        IncludeModel(
            name="fuel_env_urban_platform_shell",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/urban%20platform",
            x=0.866099,
            y=8.800140,
            z=-0.566107,
            yaw=1.57,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_r1",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=5.379010,
            y=2.308660,
            z=0.167325,
            yaw=-0.837659,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_l1",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=-4.845950,
            y=4.073690,
            z=0.0,
            yaw=0.296836,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_r2",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=-0.409183,
            y=6.872600,
            z=-3.328980,
            yaw=-1.223260,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_l2",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=-4.001130,
            y=13.511200,
            z=0.0,
            yaw=0.820759,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_r3",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=-1.559780,
            y=22.560100,
            z=0.0,
            yaw=-1.10,
        ),
        IncludeModel(
            name="fuel_env_tunnel_collapse_l3",
            uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/tunnel%20wall%20debris",
            x=-3.780870,
            y=-2.615280,
            z=0.229140,
            yaw=1.10,
        ),
    ]


def build_world() -> str:
    concrete = "0.56 0.55 0.53 1"
    slab = "0.44 0.43 0.41 1"
    rust = "0.42 0.31 0.22 1"
    ash = "0.18 0.18 0.18 1"
    caution = "0.95 0.78 0.12 1"
    beacon = "0.15 0.85 0.35 1"
    victim = "0.82 0.15 0.15 1"

    lines = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.10">',
        '  <world name="rescue_ruins_world">',
        '    <gravity>0 0 -9.8</gravity>',
        '    <physics name="1ms" type="ignored">',
        '      <max_step_size>0.002</max_step_size>',
        '      <real_time_factor>1.0</real_time_factor>',
        '    </physics>',
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        '    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"/>',
        '    <scene><ambient>0.38 0.36 0.34 1</ambient><background>0.72 0.71 0.68 1</background></scene>',
        '    <light name="sun" type="directional">',
        '      <cast_shadows>true</cast_shadows>',
        '      <pose>0 0 30 0 0 0</pose>',
        '      <diffuse>0.82 0.79 0.74 1</diffuse>',
        '      <specular>0.22 0.22 0.22 1</specular>',
        '      <direction>-0.35 0.18 -0.92</direction>',
        '    </light>',
        '    <model name="ground_plane">',
        '      <static>true</static>',
        '      <link name="ground_link">',
        '        <collision name="collision">',
        '          <geometry><plane><normal>0 0 1</normal><size>120 120</size></plane></geometry>',
        '        </collision>',
        '        <visual name="visual">',
        '          <geometry><plane><normal>0 0 1</normal><size>120 120</size></plane></geometry>',
        '          <material><ambient>0.46 0.44 0.42 1</ambient><diffuse>0.46 0.44 0.42 1</diffuse></material>',
        '        </visual>',
        '      </link>',
        '    </model>',
    ]

    doors = centered_doors()
    obstacles = centered_mid_obstacles()
    center_x, half_width = scene_layout_x(doors, obstacles)
    start, goal = start_goal_markers()
    boxes = []
    boxes.extend(corridor_shell_boxes(center_x, half_width))
    boxes.extend(obstacle_ruin_boxes(obstacles))
    boxes.extend(side_rubble_boxes(center_x, half_width))
    boxes.extend(decal_boxes(center_x, half_width))
    boxes.extend(
        [
            BoxModel("fire_hazard_1", 0.72, 0.72, 0.08, center_x + (half_width - 0.28), 5.2, 0.04, color="0.35 0.08 0.02 0.35"),
            BoxModel("fire_hazard_2", 0.76, 0.76, 0.08, center_x - (half_width - 0.26), 12.9, 0.04, color="0.35 0.08 0.02 0.35"),
        ]
    )

    for box in boxes:
        add_box_model(lines, box)

    for door in doors:
        for box in doorway_ruin_boxes(door):
            add_box_model(lines, box)
    add_doorway_lights(lines, doors)
    add_rotating_goal_cross_model(
        lines,
        name="goal_cross_marker",
        x=float(goal[0]),
        y=float(goal[1]),
        z=0.95,
    )
    flames = [
        CylinderModel("fire_core_1", 0.16, 0.36, center_x + (half_width - 0.26), 5.2, 0.20, "1.0 0.45 0.08 1"),
        CylinderModel("smoke_column_1", 0.28, 0.78, center_x + (half_width - 0.26), 5.2, 0.76, "0.16 0.16 0.16 0.65"),
        CylinderModel("fire_core_2", 0.16, 0.40, center_x - (half_width - 0.24), 12.9, 0.22, "1.0 0.40 0.06 1"),
        CylinderModel("smoke_column_2", 0.30, 0.82, center_x - (half_width - 0.24), 12.9, 0.80, "0.16 0.16 0.16 0.65"),
        CylinderModel("safe_start_ring", 0.42, 0.08, float(start[0]), float(start[1] + 0.2), 0.04, beacon),
    ]

    for cyl in flames:
        add_cylinder_model(lines, cyl)

    for inc in environment_includes(center_x, half_width):
        add_include_model(lines, inc)

    add_point_light(lines, "fire_light_1", center_x + (half_width - 0.26), 5.2, 0.95)
    add_point_light(lines, "fire_light_2", center_x - (half_width - 0.24), 12.9, 0.98)

    lines.extend(
        [
            '  </world>',
            '</sdf>',
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    doors = centered_doors()
    obstacles = centered_mid_obstacles()
    ensure_ruins_meshes()
    ensure_door_meshes(doors)
    ensure_obstacle_meshes(obstacles)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WORLD_DIR / "rescue_ruins_world.sdf"
    out_path.write_text(build_world(), encoding="ascii")
    print(f"Generated world file: {out_path}")


if __name__ == "__main__":
    main()
