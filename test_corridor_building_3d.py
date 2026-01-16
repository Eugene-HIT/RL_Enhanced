import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def _norm(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    if n < eps:
        return 0.0
    return n

def _unit(v, eps=1e-12):
    n = _norm(v, eps)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def aabb_intersect(a, b) -> bool:
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax0 <= bx1 and ax1 >= bx0 and
            ay0 <= by1 and ay1 >= by0 and
            az0 <= bz1 and az1 >= bz0)

def draw_aabb_proj(ax, bounds, plane="xy", **kwargs):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if plane == "xy":
        x0, x1 = xmin, xmax
        y0, y1 = ymin, ymax
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, **kwargs)
    elif plane == "yz":
        x0, x1 = ymin, ymax
        y0, y1 = zmin, zmax
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, **kwargs)
    else:
        raise ValueError("plane must be 'xy' or 'yz'")
    ax.add_patch(rect)

# ============================================================
# OBB Tube Corridor (per-segment, aligned with segment direction)
#
# Each segment corridor is an oriented box:
#   center c = (p0+p1)/2
#   axis d   = unit(p1-p0)   (long axis)
#   axis u   = unit(cross(up, d))   (side axis in horizontal plane)
#   axis v   = cross(d, u)          (the remaining axis)
# half-length hl = 0.5*|p1-p0| + pad_long
# radii:
#   r_xy : extent along u (tube width in the horizontal plane)
#   r_z  : extent along vertical axis (we will use global z to visualize YZ nicely)
#
# NOTE:
# - For a true 3D OBB, the cross-section is u-v, not (horizontal, vertical) strictly.
# - Here we keep visualization intuitive:
#   XY uses +/- r_xy along u
#   YZ uses +/- r_z along global z
# ============================================================

def build_tube_corridor_obbs(
    pts: np.ndarray,
    r_xy: float = 0.45,
    r_z: float = 0.35,
    pad_long: float = 0.10,
):
    pts = np.asarray(pts, float)
    assert pts.ndim == 2 and pts.shape[1] == 3
    obbs = []

    up_candidates = [
        np.array([0.0, 0.0, 1.0]),  # prefer z-up
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg = p1 - p0
        L = _norm(seg)

        if L < 1e-9:
            # degenerate segment: make a tiny box around p0
            d = np.array([1.0, 0.0, 0.0])
            u = np.array([0.0, 1.0, 0.0])
            hl = pad_long
            c = p0.copy()
        else:
            d = _unit(seg)
            # pick an "up" not parallel to d
            up = None
            for cand in up_candidates:
                if _norm(np.cross(cand, d)) > 1e-6:
                    up = cand
                    break
            if up is None:
                up = np.array([0.0, 0.0, 1.0])

            u = _unit(np.cross(up, d))   # side axis
            hl = 0.5 * L + pad_long
            c = 0.5 * (p0 + p1)

        obbs.append({
            "i": i,
            "p0": p0,
            "p1": p1,
            "c": c,
            "d": d,
            "u": u,
            "hl": float(hl),
            "r_xy": float(r_xy),
            "r_z": float(r_z),
        })
    return obbs

def obb_to_halfspaces_for_qp(obb):
    """
    Convert the tube OBB to linear halfspaces A p <= b (in 3D) for ONE segment.

    We enforce:
      |d·(p-c)| <= hl
      |u·(p-c)| <= r_xy
      |ez·(p-c)| <= r_z     (use global z as vertical bound, simple & stable)

    This is 6 inequalities:
      +a^T p <= a^T c + bound
      -a^T p <= -a^T c + bound
    """
    c = obb["c"]
    d = obb["d"]
    u = obb["u"]
    ez = np.array([0.0, 0.0, 1.0])

    hl  = obb["hl"]
    rxy = obb["r_xy"]
    rz  = obb["r_z"]

    axes = [(d, hl), (u, rxy), (ez, rz)]
    A_list = []
    b_list = []

    for a, bound in axes:
        a = np.asarray(a, float).reshape(3)
        # +a
        A_list.append(+a)
        b_list.append(float(a @ c + bound))
        # -a
        A_list.append(-a)
        b_list.append(float((-a) @ c + bound))

    A = np.vstack(A_list)  # (6,3)
    b = np.array(b_list)   # (6,)
    return A, b

def draw_obb_proj(ax, obb, plane="xy", **kwargs):
    """
    Draw oriented rectangle projection of the OBB tube.
    XY: uses +/- hl*d +/- r_xy*u (take x,y)
    YZ: uses +/- hl*d +/- r_z*ez (take y,z)  (intuitive vertical thickness)
    """
    c = obb["c"]
    d = obb["d"]
    hl = obb["hl"]

    def rect_from_center_dir(center2, dir2, half_long, half_short):
        t = np.asarray(dir2, float)
        tn = np.linalg.norm(t)
        if tn < 1e-8:
            t = np.array([1.0, 0.0], float)
            tn = 1.0
        t = t / tn
        # perpendicular (rotate 90 deg)
        n = np.array([-t[1], t[0]], float)
        return np.vstack([
            center2 + half_long * t + half_short * n,
            center2 + half_long * t - half_short * n,
            center2 - half_long * t - half_short * n,
            center2 - half_long * t + half_short * n,
        ])

    if plane == "xy":
        dir2 = np.array([d[0], d[1]], float)
        center2 = np.array([c[0], c[1]], float)
        half_short = obb["r_xy"]
        P = rect_from_center_dir(center2, dir2, hl, half_short)

    elif plane == "yz":
        dir2 = np.array([d[1], d[2]], float)
        center2 = np.array([c[1], c[2]], float)
        half_short = obb["r_z"]
        P = rect_from_center_dir(center2, dir2, hl, half_short)

    else:
        raise ValueError("plane must be 'xy' or 'yz'")

    poly = plt.Polygon(P, fill=False, **kwargs)
    ax.add_patch(poly)

# ============================================================
# Visualization
# ============================================================

def plot_tube_corridor_and_path(
    pts: np.ndarray,
    obbs: list,
    obstacle_aabbs: list,
    out_xy="tube_corridor_xy.png",
    out_yz="tube_corridor_yz.png",
):
    pts = np.asarray(pts, float)

    # XY
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Tube Corridor (OBB) & Path (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=4, label="Path points")

    for obb in obbs:
        draw_obb_proj(ax, obb, plane="xy", linewidth=2.0, edgecolor="green", alpha=0.9)

    for obs in obstacle_aabbs:
        draw_aabb_proj(ax, obs, plane="xy", linewidth=3.0, edgecolor="black")

    ax.legend()
    ax.grid(True, alpha=0.25)
    # keep equal scaling so rectangles are not distorted
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_xy, dpi=200)
    plt.close(fig)

    # YZ
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Tube Corridor (OBB) & Path (YZ)")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.plot(pts[:, 1], pts[:, 2], "-o", markersize=4, label="Path points")

    for obb in obbs:
        draw_obb_proj(ax, obb, plane="yz", linewidth=2.0, edgecolor="green", alpha=0.9)

    for obs in obstacle_aabbs:
        draw_aabb_proj(ax, obs, plane="yz", linewidth=3.0, edgecolor="black")

    ax.legend()
    ax.grid(True, alpha=0.25)
    # keep equal scaling so rectangles are not distorted
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_yz, dpi=200)
    plt.close(fig)

# ============================================================
# Example (1 obstacle + a detour 3D path around it)
# ============================================================

def main():
    # One obstacle AABB in 3D
    obstacle = (0.8, 1.2, 3.5, 6.5, 0.6, 1.4)
    obstacle_aabbs = [obstacle]

    # A 3D point sequence around the obstacle
    pts = np.array([
        [0.2,  0.0, 0.8],
        [0.4,  1.5, 0.85],
        [0.6,  3.0, 0.90],
        [0.6,  4.0, 0.95],
        [0.6,  5.2, 1.05],
        [0.6,  6.8, 1.00],
        [1.4,  8.0, 0.95],
        [1.6, 10.0, 0.90],
    ], dtype=float)

    # Build tube corridor OBBs (aligned with each segment direction)
    obbs = build_tube_corridor_obbs(
        pts,
        r_xy=0.45,
        r_z=0.35,
        pad_long=0.10,
    )

    # Print a couple of halfspaces (for later QP integration)
    A0, b0 = obb_to_halfspaces_for_qp(obbs[0])
    print("[halfspace example] seg0 A shape:", A0.shape, "b shape:", b0.shape)
    print("A0=\n", np.array2string(A0, precision=3, suppress_small=True))
    print("b0=\n", np.array2string(b0, precision=3, suppress_small=True))

    # Plot XY & YZ
    plot_tube_corridor_and_path(
        pts, obbs, obstacle_aabbs,
        out_xy="tube_corridor_xy.png",
        out_yz="tube_corridor_yz.png",
    )
    print("[viz] saved: tube_corridor_xy.png, tube_corridor_yz.png")

if __name__ == "__main__":
    main()
