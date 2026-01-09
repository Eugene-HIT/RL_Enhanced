# geom_random.py
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, scale
from shapely.geometry import Polygon, MultiPolygon

def ensure_single_polygon(geom):
    """
    保证返回的是 Polygon
    - Polygon: 直接返回
    - MultiPolygon: 取面积最大的那个
    - 其他情况: 返回 None
    """
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            return None
        return max(geom.geoms, key=lambda g: g.area)
    return None

def random_star_shaped_polygon(
    rng: np.random.Generator,
    n_vertices: int = 10,
    radius_mean: float = 0.6,
    radius_jitter: float = 0.25,
    angle_jitter: float = 0.25,
):
    """
    生成一个“星形（star-shaped）”简单多边形：按角度排序 + 随机半径。
    这种方法非常轻量，天然不自交（基本上），并且容易产生凹凸变化。
    返回：shapely Polygon（以原点为中心）
    """
    n = int(n_vertices)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # 给角度加抖动，避免规则多边形
    angles = angles + rng.uniform(-angle_jitter, angle_jitter, size=n)
    angles = np.mod(angles, 2*np.pi)
    angles.sort()

    # 半径随机：产生凹凸
    radii = radius_mean * (1.0 + rng.uniform(-radius_jitter, radius_jitter, size=n))
    radii = np.clip(radii, 0.05, None)

    pts = np.stack([radii*np.cos(angles), radii*np.sin(angles)], axis=1)
    poly = Polygon(pts).buffer(0.0)  # 修复可能的轻微数值问题
    return poly


def apply_random_notches(
    rng: np.random.Generator,
    poly: Polygon,
    n_notches: int = 2,
    notch_depth_range=(0.15, 0.45),
    notch_width_range=(0.20, 0.55),
):
    """
    在多边形上“制造凹陷”：用“挖一个矩形缺口”的方式近似。
    注意：我们不是做布尔差集（会变慢/复杂），而是直接在边界上插入凹点。
    为简单起见，我们在 bbox 的四个边之一制造 notch。
    """
    # 取 bbox
    xmin, ymin, xmax, ymax = poly.bounds
    w = xmax - xmin
    h = ymax - ymin
    if w < 1e-6 or h < 1e-6:
        return poly

    pts = np.array(poly.exterior.coords[:-1], dtype=float)  # (N,2)

    for _ in range(int(n_notches)):
        side = rng.integers(0, 4)  # 0:top 1:bottom 2:left 3:right
        depth = rng.uniform(*notch_depth_range)
        width = rng.uniform(*notch_width_range)

        # notch 尺寸按 bbox 尺度缩放（更稳定）
        d = depth * min(w, h)
        ww = width * min(w, h)

        if side in [0, 1]:
            # top/bottom: notch along x
            x0 = rng.uniform(xmin + 0.2*w, xmax - 0.2*w)
            xa = x0 - ww/2
            xb = x0 + ww/2
            xa = np.clip(xa, xmin + 0.05*w, xmax - 0.05*w)
            xb = np.clip(xb, xmin + 0.05*w, xmax - 0.05*w)

            if side == 0:  # top notch downward
                y_edge = ymax
                y_in = ymax - d
                notch = np.array([[xa, y_edge], [xa, y_in], [xb, y_in], [xb, y_edge]])
            else:          # bottom notch upward
                y_edge = ymin
                y_in = ymin + d
                notch = np.array([[xb, y_edge], [xb, y_in], [xa, y_in], [xa, y_edge]])
        else:
            # left/right: notch along y
            y0 = rng.uniform(ymin + 0.2*h, ymax - 0.2*h)
            ya = y0 - ww/2
            yb = y0 + ww/2
            ya = np.clip(ya, ymin + 0.05*h, ymax - 0.05*h)
            yb = np.clip(yb, ymin + 0.05*h, ymax - 0.05*h)

            if side == 2:  # left notch to right
                x_edge = xmin
                x_in = xmin + d
                notch = np.array([[x_edge, ya], [x_in, ya], [x_in, yb], [x_edge, yb]])
            else:          # right notch to left
                x_edge = xmax
                x_in = xmax - d
                notch = np.array([[x_edge, yb], [x_in, yb], [x_in, ya], [x_edge, ya]])

        # 关键：用“把 notch 四个点插入到最近的边附近”的方式制造凹陷（轻量）
        # 找到 pts 中最接近 notch 第一个点的位置插入
        p0 = notch[0]
        idx = int(np.argmin(np.sum((pts - p0)**2, axis=1)))
        pts = np.insert(pts, idx+1, notch, axis=0)

        # 重新成 polygon 并修复
        poly2 = Polygon(pts).buffer(0.0)
        poly2 = ensure_single_polygon(poly2)

        if poly2 is not None and poly2.is_valid and poly2.area > 1e-4:
            poly = poly2
            pts = np.array(poly.exterior.coords[:-1], dtype=float)

    return poly


def sample_random_door_polygon(
    seed: int = None,
    center=(1.2, 1.0),
    scale_x_range=(0.9, 1.8),
    scale_z_range=(0.9, 1.8),
    n_vertices_range=(8, 16),
    concave_prob=0.7,
    n_notches_range=(0, 3),
):
    """
    最终对外接口：生成一个随机门洞多边形（凹凸随机），并平移到 center 附近
    返回：
      poly: shapely Polygon (global)
      verts: (N,2) 顶点
    """
    rng = np.random.default_rng(seed)

    n_vertices = int(rng.integers(n_vertices_range[0], n_vertices_range[1] + 1))
    base = random_star_shaped_polygon(
        rng,
        n_vertices=n_vertices,
        radius_mean=0.6,
        radius_jitter=0.35,
        angle_jitter=0.35,
    )

    # 随机制造凹陷（notches）
    if rng.random() < concave_prob:
        n_notches = int(rng.integers(n_notches_range[0], n_notches_range[1] + 1))
        base = apply_random_notches(rng, base, n_notches=n_notches)

    # 缩放到你环境的尺度
    sx = float(rng.uniform(*scale_x_range))
    sz = float(rng.uniform(*scale_z_range))
    base = scale(base, xfact=sx, yfact=sz, origin=(0, 0))

    # 平移到指定中心
    cx, cz = center
    base = translate(base, xoff=cx, yoff=cz)

    verts = np.array(base.exterior.coords[:-1], dtype=float)
    return base, verts
