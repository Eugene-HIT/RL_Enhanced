# geom_encoder.py
import numpy as np
from shapely.geometry import Polygon, Point

def _as_valid_polygon(poly_or_pts):
    """输入可以是 Polygon 或 (N,2) 点集。输出一个修复后的 Polygon。"""
    if isinstance(poly_or_pts, Polygon):
        poly = poly_or_pts
    else:
        pts = np.asarray(poly_or_pts, dtype=float)
        poly = Polygon(pts)
    # 修复自交/数值问题
    poly = poly.buffer(0.0)
    if poly.is_empty:
        raise ValueError("Invalid polygon: empty after buffer(0).")
    return poly

def polygon_to_grid(poly_or_pts, n=16, samples=3, return_meta=True):
    """
    把门洞多边形编码成 n×n grid（每格为 p_free ∈ [0,1] 的占比）。
    - grid 是在 bbox 局部归一化坐标下均匀采样得到的，但输出是固定 n×n。
    - center 默认用 bbox center（更稳定，尤其凹形）。

    Returns:
      grid: (n,n) float32, p_free
      center: (2,) float32, (cx,cz) in global
      bbox: (xmin,zmin,xmax,zmax) float
      meta: dict (optional)
    """
    poly = _as_valid_polygon(poly_or_pts)

    xmin, zmin, xmax, zmax = poly.bounds
    w = max(xmax - xmin, 1e-9)
    h = max(zmax - zmin, 1e-9)
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)

    # cell size in global units
    dx = w / n
    dz = h / n

    # 在每个 cell 内做 samples×samples 采样
    s = int(samples)
    # 采样点相对 cell 左下角的偏移（均匀分布在 (0,1) 内）
    offs = (np.arange(s, dtype=float) + 0.5) / s

    grid = np.zeros((n, n), dtype=np.float32)
    for iz in range(n):
        for ix in range(n):
            x0 = xmin + ix * dx
            z0 = zmin + iz * dz

            inside = 0
            total = s * s
            for oz in offs:
                for ox in offs:
                    x = x0 + ox * dx
                    z = z0 + oz * dz
                    if poly.contains(Point(x, z)):
                        inside += 1
            p_free = inside / total
            grid[iz, ix] = p_free  # 注意这里 iz 对应 z 方向

    meta = None
    if return_meta:
        meta = {
            "door_w": float(w),
            "door_h": float(h),
            "res_x": float(dx),
            "res_z": float(dz),
        }

    return grid, np.array([cx, cz], dtype=np.float32), (xmin, zmin, xmax, zmax), meta
