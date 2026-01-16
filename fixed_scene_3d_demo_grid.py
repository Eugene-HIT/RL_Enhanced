import os
import numpy as np
import matplotlib
from scipy.io import savemat

# 通过环境变量控制是否弹出交互窗口：SET SHOW_PLOTS=1 即可展示 plot 窗口
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") in ("1", "true", "True")
if not SHOW_PLOTS:
    matplotlib.use('Agg')  # 默认仍用无界面后端
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from stable_baselines3 import SAC
from shapely.geometry import LineString
from traj_qp_corridor import (
    solve_minsnap_xyz_with_box_corridors_osqp,
    build_box_corridor_constraints,
    visualize_box_corridor_and_trajectory,
    check_traj_in_box_corridor,
    visualize_keyframes_detail,
)

import torch
print(f"[init][torch] version={torch.__version__}")

from env_passenv_grid import PassEnvGrid
from geom_encoder import polygon_to_grid
import heapq
from shapely.geometry import Point

def corridor_poly_for_segment_xz(a, b, r=0.25):
    seg = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])
    poly = seg.buffer(r)
    return poly.convex_hull

def convex_polygon_to_halfspaces(poly):
    """
    输入 shapely Polygon（假设凸），输出 A,b 使得 A @ [x,z] <= b 表示内部。
    """
    coords = np.asarray(poly.exterior.coords[:-1], dtype=float)  # (M,2)
    # 判断顶点顺序（CCW or CW）
    area2 = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i+1) % len(coords)]
        area2 += (x1*y2 - x2*y1)
    ccw = area2 > 0

    A = []
    b = []
    for i in range(len(coords)):
        p = coords[i]
        q = coords[(i+1) % len(coords)]
        e = q - p  # edge direction
        # inward normal
        if ccw:
            n = np.array([ e[1], -e[0] ], dtype=float)
        else:
            n = np.array([ -e[1], e[0] ], dtype=float)
        # normalize (not required, but helps conditioning)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        A.append(n)
        b.append(n @ p)
    return np.vstack(A), np.asarray(b)

def visualize_astar_debug_xy(doors, forbidden_prisms, keyframes_P, out_path="astar_debug_xy.png"):
    """
    画 XY 平面调试图：
    - 障碍物的中心点和 XY 平面轮廓（y 区间矩形）
    - keyframes 的 (x,y) 点和连线
    - 禁飞区的 y 区间（可选，半透明）
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # 1) 画障碍物（doors）的轮廓和中心点
    for door in doors:
        xmin, _, xmax, _ = door["bbox"]  # bbox 是 (xmin, zmin, xmax, zmax)
        y_min = door["y_min"]
        y_max = door["y_max"]
        cx = door["door_cx"]
        cy = 0.5 * (y_min + y_max)  # y 中间点

        # 画轮廓：y 方向的矩形跨度
        rect = patches.Rectangle((xmin, y_min), xmax - xmin, y_max - y_min,
                                 linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)

        # 画中心点
        ax.scatter([cx], [cy], s=50, c='red', marker='x', label=f"{door['name']} center" if door == doors[0] else "")

    # 1.5) 画禁飞区（forbidden_prisms）的轮廓
    for f in forbidden_prisms:
        if "poly_xz" in f:
            xmin, _, xmax, _ = f["poly_xz"].bounds
            y_min = f["y_min"]
            y_max = f["y_max"]
            rect = patches.Rectangle((xmin, y_min), xmax - xmin, y_max - y_min,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(rect)

    # 2) 画禁飞区的 y 区间（可选，半透明横条，但只在相关 x 范围？这里简化全宽）
    for f in forbidden_prisms:
        y0, y1 = f["y_min"], f["y_max"]
        ax.axhspan(y0, y1, alpha=0.05, color='gray')

    # 3) keyframes 点与连线
    xk = keyframes_P[:, 0]
    yk = keyframes_P[:, 1]
    ax.scatter(xk, yk, s=30, c='green')
    ax.plot(xk, yk, 'g-o', alpha=0.7, linewidth=2)  # 连线

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_title("XY Debug: doors centers/outlines + keyframes path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[viz][astar_xy] saved path={out_path}")


def visualize_astar_debug_xz(doors, forbidden_prisms, keyframes_P, out_path="astar_debug_xz.png"):
    """
    画 XZ 平面调试图：
    - 门洞 poly（可通行区域）
    - 禁飞区 poly_xz（不可通行区域）
    - keyframes 的 (x,z) 点
    - 相邻 keyframes 的直线连接
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # 1) doors（可通行开口区域）
    for d in doors:
        poly = d["poly"]
        x, z = poly.exterior.xy
        ax.plot(x, z, linewidth=2)
        # 标注门洞名字（用中心点）
        cx, cz = poly.centroid.x, poly.centroid.y
        ax.text(cx, cz, d["name"], fontsize=10)

    # 2) forbidden（禁飞区）
    for f in forbidden_prisms:
        poly = f["poly_xz"]
        geoms = getattr(poly, "geoms", [poly])
        for g in geoms:
            if g.is_empty:
                continue
            x, z = g.exterior.xy
            ax.plot(x, z, linestyle="--", linewidth=2)

    # 3) keyframes 点与连线
    xk = keyframes_P[:, 0]
    zk = keyframes_P[:, 2]
    ax.scatter(xk, zk, s=25)
    for i in range(len(keyframes_P) - 1):
        a = keyframes_P[i]
        b = keyframes_P[i + 1]
        ax.plot([a[0], b[0]], [a[2], b[2]], alpha=0.6)

        # 标注点序号（防止你看不懂哪段是门间段）
        ax.text(a[0], a[2], f"{i}", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("XZ Debug: doors(solid) + forbidden(dashed) + keyframes(points/lines)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[viz][astar_xz] saved path={out_path}")
    
def visualize_astar_debug_xz(doors, forbidden_prisms, keyframes_P, out_path="astar_debug_xz.png"):
    """
    画 XZ 平面调试图：
    - 门洞 poly（可通行区域）
    - 禁飞区 poly_xz（不可通行区域）
    - keyframes 的 (x,z) 点
    - 相邻 keyframes 的直线连接
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # 1) doors（可通行开口区域）
    for d in doors:
        poly = d["poly"]
        x, z = poly.exterior.xy
        ax.plot(x, z, linewidth=2)
        # 标注门洞名字（用中心点）
        cx, cz = poly.centroid.x, poly.centroid.y
        ax.text(cx, cz, d["name"], fontsize=10)

    # 2) forbidden（禁飞区）
    for f in forbidden_prisms:
        poly = f["poly_xz"]
        geoms = getattr(poly, "geoms", [poly])
        for g in geoms:
            if g.is_empty:
                continue
            x, z = g.exterior.xy
            ax.plot(x, z, linestyle="--", linewidth=2)

    # 3) keyframes 点与连线
    xk = keyframes_P[:, 0]
    zk = keyframes_P[:, 2]
    ax.scatter(xk, zk, s=25)
    for i in range(len(keyframes_P) - 1):
        a = keyframes_P[i]
        b = keyframes_P[i + 1]
        ax.plot([a[0], b[0]], [a[2], b[2]], alpha=0.6)

        # 标注点序号（防止你看不懂哪段是门间段）
        ax.text(a[0], a[2], f"{i}", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("XZ Debug: doors(solid) + forbidden(dashed) + keyframes(points/lines)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[viz][astar_xz] saved path={out_path}")

def debug_segment_intersections(P_wp, forbidden_prisms, seg_name=""):
    """
    打印每段相邻 keyframe 的 XZ 线段是否与禁飞区相交 + y 区间是否重叠
    """
    print(f"\n[debug][segment_intersections] name={seg_name}")
    for i in range(len(P_wp) - 1):
        a = P_wp[i]; b = P_wp[i+1]
        line_xz = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])

        hit_any = False
        hit_tags = []
        for f in forbidden_prisms:
            # y 方向是否可能相交（区间重叠）
            ymin = min(float(a[1]), float(b[1]))
            ymax = max(float(a[1]), float(b[1]))
            y_overlap = not (ymax < f["y_min"] or ymin > f["y_max"])

            if not y_overlap:
                continue

            poly = f["poly_xz"]
            if line_xz.intersects(poly):
                hit_any = True
                hit_tags.append(f.get("tag", "forbidden"))

        if hit_any:
            print(f"[debug][segment_hit] seg={i}->{i+1} tags={hit_tags}")

def make_forbidden_prisms_from_doors(doors, margin=0.12, y_pad=0.05, y_extend=1.2):
    """
    由门洞构造禁飞区：
    1) 门框环带：expanded - door_poly，作用于 [y_min-y_pad, y_max+y_pad]
    2) 门前墙体片：同样的环带，作用于 [y_min - y_extend, y_min]
    3) 门后墙体片：同样的环带，作用于 [y_max, y_max + y_extend]
    """
    forbidden = []
    for door in doors:
        door_poly = door["poly"].buffer(0.0)
        expanded = door_poly.buffer(margin)
        ring = expanded.difference(door_poly)

        geoms = getattr(ring, "geoms", [ring])
        for g in geoms:
            if g.is_empty or g.area < 1e-6:
                continue
            gxz = g.buffer(0.0)

            y0 = float(door["y_min"])
            y1 = float(door["y_max"])

            # (A) 门框本体
            forbidden.append({
                "poly_xz": gxz,
                "y_min": y0 - y_pad,
                "y_max": y1 + y_pad,
                "tag": f"{door['name']}_frame",
            })

            # (B) 门前墙体片（门外前方）
            forbidden.append({
                "poly_xz": gxz,
                "y_min": y0 - y_extend,
                "y_max": y0,
                "tag": f"{door['name']}_front",
            })

            # (C) 门后墙体片（门外后方）
            forbidden.append({
                "poly_xz": gxz,
                "y_min": y1,
                "y_max": y1 + y_extend,
                "tag": f"{door['name']}_back",
            })

    return forbidden

    """
    用“门洞拓张环带”构造禁飞区（门框/墙体带）。
    - margin: 门洞外扩厚度（越大表示墙越厚/安全裕度越大）
    - y_pad : y 方向额外扩展（让禁飞区在门洞前后稍微延伸）
    返回：list of prisms, each prism has poly_xz + y_min/y_max
    """
    forbidden = []
    for door in doors:
        door_poly = door["poly"].buffer(0.0)

        # 外扩门洞
        expanded = door_poly.buffer(margin)

        # 环带 = expanded - door_poly：代表“门框/墙体区域”
        ring = expanded.difference(door_poly)

        # ring 可能是 Polygon 或 MultiPolygon，统一拆成多个 Polygon
        geoms = getattr(ring, "geoms", [ring])
        for g in geoms:
            if g.is_empty:
                continue
            # 过滤很小碎片
            if g.area < 1e-6:
                continue

            forbidden.append({
                "poly_xz": g.buffer(0.0),
                "y_min": float(door["y_min"] - y_pad),
                "y_max": float(door["y_max"] + y_pad),
                "tag": door["name"],
            })

    return forbidden

def build_keyframes_with_astar(doors, sols, start, goal, forbidden_prisms,
                               pre_dist=0.8, post_dist=0.8,
                               corridor_radius=1.2, corridor_safety=0.30,
                               debug_keyframes=False,
                               return_astar_paths=False):
    """
    Return:
      P: (N,3) keyframe positions
      R: (N,)  roll/theta reference (currently piecewise-constant like original)
      tags: (N,) str tags in {"start","door_pre","door_mid","door_post","astar","goal"}
      hard: (N,) bool, True for pre/mid/post (+ start/goal by default)
      door_ids: (N,) int, -1 for start/goal/astar, otherwise door index
    """
    P = [np.asarray(start, float)]
    R = [0.0]
    tags = ["start"]
    hard = [True]          # start 是否硬约束：建议 True（QP更稳定）
    door_ids = [-1]
    astar_paths = []

    triplets = []
    for door, sol in zip(doors, sols):
        y0 = float(door["y_min"])
        y1 = float(door["y_max"])
        ym = 0.5 * (y0 + y1)
        cx, cz, th = float(sol["cx"]), float(sol["cz"]), float(sol["theta"])

        pre  = np.array([cx, y0 - pre_dist,  cz], float)
        mid  = np.array([cx, ym,             cz], float)
        post = np.array([cx, y1 + post_dist, cz], float)
        triplets.append((pre, mid, post, th))

        if debug_keyframes:
            print(f"[door][y] y_range=[{y0:.2f},{y1:.2f}] thick={y1-y0:.2f} pre={pre[1]:.2f} mid={mid[1]:.2f} post={post[1]:.2f}")

    # start -> door0 pre
    P.append(triplets[0][0]); R.append(triplets[0][3])
    tags.append("door_pre"); hard.append(True); door_ids.append(0)

    for i in range(len(triplets)):
        pre, mid, post, th = triplets[i]

        # mid, post
        P.append(mid);  R.append(th)
        tags.append("door_mid");  hard.append(True); door_ids.append(i)

        P.append(post); R.append(th)
        tags.append("door_post"); hard.append(True); door_ids.append(i)

        if i < len(triplets) - 1:
            next_pre, _, _, next_th = triplets[i+1]

            seg_pts = plan_segment_astar(
                post, next_pre, forbidden_prisms,
                corridor_radius=corridor_radius,
                n_slices=80, pts_per_slice=80, knn=24, step_check=0.01,
                safety=corridor_safety
            )
            print(f"[astar] segment={i} nodes={seg_pts.shape[0]}")
            if return_astar_paths:
                astar_paths.append({
                    "segment": i,
                    "from_door": i,
                    "to_door": i + 1,
                    "points": np.asarray(seg_pts, float),
                })

            # 插中间点（避免重复端点）
            for p in seg_pts[1:-1]:
                P.append(p); R.append(th)
                tags.append("astar"); hard.append(False); door_ids.append(-1)

            # next door pre
            P.append(next_pre); R.append(next_th)
            tags.append("door_pre"); hard.append(True); door_ids.append(i+1)

    # goal
    P.append(np.asarray(goal, float)); R.append(0.0)
    tags.append("goal"); hard.append(True); door_ids.append(-1)

    result = (np.asarray(P, float),
              np.asarray(R, float),
              np.asarray(tags, dtype=object),
              np.asarray(hard, dtype=bool),
              np.asarray(door_ids, dtype=int))

    if return_astar_paths:
        return (*result, astar_paths)
    return result


def _template_polygon(shape_type: int) -> np.ndarray:
    """Copy of env templates (local coords, centered)."""
    if shape_type == 0:  # rectangle
        pts = np.array([[-0.25, -0.5],
                        [ 0.25, -0.5],
                        [ 0.25,  0.5],
                        [-0.25,  0.5]], dtype=float)
    elif shape_type == 1:  # right triangle
        pts = np.array([[-0.3, -0.5],
                        [ 0.3, -0.5],
                        [-0.3,  0.5]], dtype=float)
    elif shape_type == 2:  # irregular pentagon (convex)
        pts = np.array([[-0.25, -0.5],
                        [ 0.25, -0.4],
                        [ 0.35,  0.0],
                        [ 0.10,  0.5],
                        [-0.30,  0.3]], dtype=float)
    else:  # 3: concave C-shape
        pts = np.array([[-0.4, -0.6],
                        [ 0.4, -0.6],
                        [ 0.4, -0.3],
                        [ 0.0, -0.3],
                        [ 0.0,  0.3],
                        [ 0.4,  0.3],
                        [ 0.4,  0.6],
                        [-0.4,  0.6]], dtype=float)

    center = pts.mean(axis=0)
    return pts - center

# ========= 1) 旧版场景：直接从你 fixed_scene_3d_demo.py 复制过来 =========
def make_fixed_doors():
    """
    4 obstacles:
      - each polygon is in XZ plane
      - extruded along Y with thickness (y_min, y_max)
      - centers are deliberately NOT collinear in global space
    """
    doors = []

    # (shape_type, sx, sz, (cx, cz), (y_min, y_max))
    # y ranges adjusted so adjacent door MIDPOINTS are ~5m apart
    specs = [
    (3, 1.20, 1.70, (0.90, 1.00), (0.00, 2.00)),   # center ~1.0
    (2, 1.60, 1.50, (1.50, 1.10), (5.00, 7.00)),   # center ~6.0
    (1, 1.20, 1.55, (1.25, 0.95), (10.00, 12.00)),  # center ~11.0
    (0, 1.50, 1.80, (1.20, 1.05), (15.00, 17.00)),  # center ~16.0
    ]


    for idx, (stype, sx, sz, (cx0, cz0), (y0, y1)) in enumerate(specs, start=1):
        base = _template_polygon(stype)
        pts = np.column_stack([sx * base[:, 0], sz * base[:, 1]]) + np.array([cx0, cz0], dtype=float)
        poly = Polygon(pts).buffer(0.0)
        xmin, zmin, xmax, zmax = poly.bounds

        doors.append({
            "name": f"door{idx}",
            "shape_type": stype,
            "poly_xz": pts,
            "poly": poly,
            "bbox": (xmin, zmin, xmax, zmax),
            "door_w": float(xmax - xmin),
            "door_h": float(zmax - zmin),
            "door_cx": float(0.5 * (xmin + xmax)),
            "door_cz": float(0.5 * (zmin + zmax)),
            "y_min": float(y0),
            "y_max": float(y1),
        })

    return doors


def door_y_mid(door):
    return 0.5 * (float(door["y_min"]) + float(door["y_max"]))


# ========= 2) 把单门洞 passing 求解换成你的新接口 + theta_prev 串联 =========
def solve_passing_for_door_grid(env: PassEnvGrid, model: SAC, door_poly: Polygon, theta_prev: float):
    """
    输入：door_poly (XZ polygon), theta_prev
    输出：cx, cz, theta, info
    """
    # ---- 注入 door polygon 到 env，并构造 grid/center/bbox（照你 test.py 写法） ----
    env.door_poly = door_poly

    grid, center, bbox, meta = polygon_to_grid(
        door_poly, n=env.grid_n, samples=env.grid_samples, return_meta=True
    )

    env._grid_cached = grid.astype(np.float32)
    env.door_center = center.astype(np.float32)     # (cx, cz) in global
    env.door_bbox = bbox
    # env.shape_type 可选：只是 info 展示用，你 test.py 里也只是为了好看 :contentReference[oaicite:3]{index=3}
    # env.shape_type = 0

    # ---- 关键：theta_prev 串联（第一个门用 0，之后用上一个门的 theta） ----
    env.theta_prev = float(theta_prev)

    # 构造 obs（env 内部实现：grid_flat + center + dims + theta_prev_norm）
    obs = env._build_obs(env._grid_cached, env.door_center, env.theta_prev)

    # 推理动作
    action, _ = model.predict(obs, deterministic=True)

    # 环境 step，把 action 映射成 (cx, cz, theta) 并检查 within
    _, _, _, _, info = env.step(action)

    # 你 env 的 info 里通常会包含：cx, cz, theta, all_inside 等
    cx = float(info["cx"])
    cz = float(info["cz"])
    theta = float(info["theta"])

    return cx, cz, theta, info


# ========= 3) keyframe 构造：沿用旧版 pre/mid/post，并保留插值（先跑通） =========
def build_keyframes_from_solutions(doors, sols, start, goal,
                                  pre_dist=0.8, post_dist=0.8,
                                  n_interp=2):
    """
    doors: door list
    sols:  list of {"cx","cz","theta"} for each door
    start/goal: np.array shape (3,) position in world
    输出：P_wp (N,3), roll_wp (N,)  其中 roll_wp 用来存 passing angle 作为姿态参考
    """
    P = [np.asarray(start, float)]
    R = [0.0]

    for door, sol in zip(doors, sols):
        y0 = float(door["y_min"])
        y1 = float(door["y_max"])
        ym = 0.5 * (y0 + y1)

        cx, cz, th = float(sol["cx"]), float(sol["cz"]), float(sol["theta"])

        pre  = np.array([cx, y0 - pre_dist, cz], float)
        mid  = np.array([cx, ym,            cz], float)
        post = np.array([cx, y1 + post_dist, cz], float)

        # pre/mid/post 都用同一个 passing angle（先这样，后面再讨论姿态时间曲线）
        P += [pre, mid, post]
        R += [th,  th,  th]

        # 门与门之间：先保留旧版插值点（下一步再替换成 corridor A*）
        # 注意：这里的插值会在下一段循环里再加一次，所以这里不加。

    P.append(np.asarray(goal, float))
    R.append(0.0)

    # 这里做“门与门之间插值”的简单实现：遍历 P，把大段插点（先跑通）
    if n_interp > 0:
        P2 = [P[0]]
        R2 = [R[0]]
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            ra, rb = R[i], R[i + 1]
            # 只对“跨越较长”的段插点（避免 pre-mid-post 这种很短的段被塞满）
            seg_len = float(np.linalg.norm(b - a))
            if seg_len > 2.0:  # 这个阈值你可调
                for k in range(1, n_interp + 1):
                    t = k / (n_interp + 1)
                    P2.append((1 - t) * a + t * b)
                    R2.append((1 - t) * ra + t * rb)
            P2.append(b)
            R2.append(rb)
        P, R = P2, R2

    return np.asarray(P, float), np.asarray(R, float)


# ========= 4) 预留：下一步替换插值为“走廊 A*” =========


def _in_forbidden(p, forbidden_prisms, safety=0.30):
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    pt = Point(x, z)
    for f in forbidden_prisms:
        if y < f["y_min"] or y > f["y_max"]:
            continue
        # distance < safety 等价于“离禁飞区太近也算碰撞”
        if f["poly_xz"].distance(pt) < safety:
            return True
    return False


def _segment_free(a, b, forbidden_prisms, step=0.05, safety=0.30):
    """线段碰撞检测：用 shapely intersects，允许安全外扩。"""
    line = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])  # x,z
    for f in forbidden_prisms:
        # y 方向重叠
        if max(float(a[1]), float(b[1])) < f["y_min"] or min(float(a[1]), float(b[1])) > f["y_max"]:
            continue
        poly_buf = f["poly_xz"].buffer(safety)
        if line.intersects(poly_buf):
            return False
    return True

def _astar(nodes, neighbors, s_id, g_id):
    def h(i):
        return float(np.linalg.norm(nodes[i] - nodes[g_id]))

    openq = []
    heapq.heappush(openq, (h(s_id), 0.0, s_id))
    parent = {s_id: -1}
    best_g = {s_id: 0.0}

    while openq:
        f, g, u = heapq.heappop(openq)
        if u == g_id:
            break
        if g > best_g.get(u, 1e30) + 1e-9:
            continue
        for v, cost in neighbors[u]:
            ng = g + cost
            if ng + 1e-9 < best_g.get(v, 1e30):
                best_g[v] = ng
                parent[v] = u
                heapq.heappush(openq, (ng + h(v), ng, v))

    if g_id not in parent:
        return None

    path = []
    cur = g_id
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def plan_segment_astar(start_p, goal_p, forbidden_prisms,
                      corridor_radius=1.2,
                      n_slices=20,
                      pts_per_slice=25,
                      knn=12,
                      step_check=0.05,
                      safety=0.30,
                      max_expand=3):
    """
    走廊采样 + KNN 建图 + A*。
    返回：(M,3) 点序列，含 start 和 goal
    """
    start_p = np.asarray(start_p, float)
    goal_p  = np.asarray(goal_p, float)

    # 直线可行就直接返回
    # if _segment_free(start_p, goal_p, forbidden_prisms, step=step_check, safety=safety):
    #     return np.vstack([start_p, goal_p])

    d = goal_p - start_p
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.vstack([start_p])

    u = d / L
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, u)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    v1 = np.cross(u, tmp); v1 /= (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(u, v1);  v2 /= (np.linalg.norm(v2) + 1e-9)

    radius = corridor_radius
    for _try in range(max_expand + 1):
        nodes = [start_p, goal_p]

        # 走廊采样：沿主轴切片，每片采圆盘点
        for i in range(1, n_slices):
            t = i / n_slices
            center = start_p + t * d
            for _ in range(pts_per_slice):
                r = radius * np.sqrt(np.random.rand())
                ang = 2 * np.pi * np.random.rand()
                p = center + r * (np.cos(ang) * v1 + np.sin(ang) * v2)
                if not _in_forbidden(p, forbidden_prisms, safety=safety):
                    nodes.append(p)

        nodes = np.asarray(nodes, float)
        N = nodes.shape[0]

        # KNN 建图（N 不大，直接 O(N^2) 找近邻也可）
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            di = np.linalg.norm(nodes - nodes[i], axis=1)
            idx = np.argsort(di)[1:knn+1]
            for j in idx:
                a, b = nodes[i], nodes[j]
                if _segment_free(a, b, forbidden_prisms, step=step_check, safety=safety):
                    cost = float(np.linalg.norm(b - a))
                    neighbors[i].append((j, cost))
                    neighbors[j].append((i, cost))

        path_ids = _astar(nodes, neighbors, 0, 1)
        if path_ids is not None:
            return nodes[path_ids]

        # 找不到路就扩走廊半径再试
        radius *= 1.5

    # 实在失败退化直线（你也可以改成 raise）
    return np.vstack([start_p, goal_p])



def _make_payload_rect(env) -> np.ndarray:
    """
    生成载荷在自身坐标系(L帧)下的矩形顶点 (x,z)。
    尽量从 env 里读尺寸；读不到就给一个明确报错提示你改变量名。
    """
    # 你 env 里一般会有有效尺寸（名字可能不同），这里做兼容读取
    cand = [
        ("L_eff", "H_eff"),
        ("payload_L", "payload_H"),
        ("payload_len", "payload_h"),
        ("load_L", "load_H"),
    ]
    L = H = None
    for a, b in cand:
        if hasattr(env, a) and hasattr(env, b):
            L = float(getattr(env, a))
            H = float(getattr(env, b))
            break

    if L is None or H is None:
        raise RuntimeError(
            "无法从 env 读取载荷尺寸，请在 _make_payload_rect() 里把变量名改成你 env 真实字段名。"
        )

    # 以质心为原点的矩形（x 方向长度 L，z 方向高度 H）
    rect = np.array([
        [-0.5 * L, -0.5 * H],
        [ 0.5 * L, -0.5 * H],
        [ 0.5 * L,  0.5 * H],
        [-0.5 * L,  0.5 * H],
    ], dtype=float)
    return rect


def _transform_polygon_xz(pts_xz: np.ndarray, cx: float, cz: float, theta: float) -> np.ndarray:
    """
    pts_xz: (N,2) in local frame
    输出：旋转+平移后的 (N,2) in world XZ
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    out = (pts_xz @ R.T) + np.array([cx, cz], dtype=float)
    return out


def visualize_passing_xz(doors, sols, env, out_path="passing_xz_debug.png"):
    """
    每个门洞画一张 XZ 截面：door poly + 载荷变换后矩形。
    并标注 theta(deg) 和 clearance(载荷到门边界最小距离，单位同坐标)。
    """
    rect_local = _make_payload_rect(env)

    n = len(doors)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (door, sol) in enumerate(zip(doors, sols)):
        ax = axes[i]
        door_poly = door["poly"]

        cx, cz, th = float(sol["cx"]), float(sol["cz"]), float(sol["theta"])
        rect_world = _transform_polygon_xz(rect_local, cx, cz, th)
        payload_poly = Polygon(rect_world).buffer(0.0)

        # door
        x_d, z_d = door_poly.exterior.xy
        ax.plot(x_d, z_d, linewidth=2)

        # payload
        x_p, z_p = payload_poly.exterior.xy
        ax.plot(x_p, z_p, linewidth=2)

        # clearance：载荷到门洞边界的最小距离（inside=True 时这个数越大越安全）
        # 用 door 边界到 payload 的距离（payload 全在内时为正）
        clearance = door_poly.exterior.distance(payload_poly)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        deg = th * 180.0 / np.pi
        ax.set_title(
            f"{door['name']} | theta={deg:+.1f} deg | clearance={clearance:.3f}",
            fontsize=12
        )

        # 画中心点
        ax.scatter([cx], [cz], s=40)
        ax.text(cx, cz, f"  ({cx:.2f},{cz:.2f})", fontsize=10)

        # 给一点边界余量
        xmin, zmin, xmax, zmax = door_poly.bounds
        pad = 0.6
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(zmin - pad, zmax + pad)

    # 多余子图关掉
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[viz][passing_xz] saved path={out_path}")


def _compute_obb_corners(center, axis_d, axis_u, hl, r_xy, r_z):
    c = np.asarray(center, float).reshape(3)
    d = np.asarray(axis_d, float).reshape(3)
    u = np.asarray(axis_u, float).reshape(3)
    w = np.array([0.0, 0.0, 1.0], float)
    corners = []
    for sd in (-1.0, 1.0):
        for su in (-1.0, 1.0):
            for sw in (-1.0, 1.0):
                corners.append(c + sd * float(hl) * d + su * float(r_xy) * u + sw * float(r_z) * w)
    return np.asarray(corners, float)


def export_to_matlab(out_path,
                     qp3d,
                     P_wp,
                     roll_wp,
                     tags,
                     hard,
                     door_ids,
                     doors,
                     forbidden,
                     box_segments,
                     astar_paths):
    """Export trajectory/scene data to a MATLAB .mat file for post-analysis."""

    def polygon_coords(poly):
        try:
            return np.asarray(poly.exterior.coords, dtype=float)
        except Exception:
            return np.empty((0, 2), dtype=float)

    doors_out = []
    for d in doors:
        poly_xz = d.get("poly_xz", None)
        if poly_xz is None:
            poly_arr = polygon_coords(d.get("poly", None))
        else:
            poly_arr = np.asarray(poly_xz, float)

        doors_out.append({
            "name": d.get("name", ""),
            "poly_xz": poly_arr,
            "bbox": np.asarray(d.get("bbox", ()), float),
            "y_min": float(d.get("y_min", 0.0)),
            "y_max": float(d.get("y_max", 0.0)),
            "door_cx": float(d.get("door_cx", 0.0)),
            "door_cz": float(d.get("door_cz", 0.0)),
        })

    forb_out = []
    for f in forbidden:
        forb_out.append({
            "tag": f.get("tag", ""),
            "poly_xz": polygon_coords(f.get("poly_xz", None)),
            "y_min": float(f.get("y_min", 0.0)),
            "y_max": float(f.get("y_max", 0.0)),
        })

    seg_out = []
    for sid, seg in enumerate(box_segments):
        poly_arr = polygon_coords(seg.get("poly_xz", None))
        A_xz = seg.get("A_xz", None)
        b_xz = seg.get("b_xz", None)
        A_obb = seg.get("A_obb", None)
        b_obb = seg.get("b_obb", None)
        corners = _compute_obb_corners(seg.get("center", [0, 0, 0]),
                                        seg.get("axis_d", [1, 0, 0]),
                                        seg.get("axis_u", [0, 1, 0]),
                                        seg.get("hl", 0.0),
                                        seg.get("r_xy", 0.0),
                                        seg.get("r_z", 0.0))

        seg_out.append({
            "id": int(sid),
            "enabled": bool(seg.get("enabled", True)),
            "idx_range": np.asarray(seg.get("idx_range", (0, 0)), int),
            "y_bounds": np.asarray(seg.get("y_bounds", (0.0, 0.0)), float),
            "bounds": np.asarray(seg.get("bounds", (0, 0, 0, 0, 0, 0)), float),
            "center": np.asarray(seg.get("center", (0, 0, 0)), float),
            "axis_d": np.asarray(seg.get("axis_d", (1, 0, 0)), float),
            "axis_u": np.asarray(seg.get("axis_u", (0, 1, 0)), float),
            "hl": float(seg.get("hl", 0.0)),
            "r_xy": float(seg.get("r_xy", 0.0)),
            "r_z": float(seg.get("r_z", 0.0)),
            "A_xz": np.asarray(A_xz, float) if A_xz is not None else np.empty((0, 2), dtype=float),
            "b_xz": np.asarray(b_xz, float) if b_xz is not None else np.empty((0,), dtype=float),
            "A_obb": np.asarray(A_obb, float) if A_obb is not None else np.empty((0, 3), dtype=float),
            "b_obb": np.asarray(b_obb, float) if b_obb is not None else np.empty((0,), dtype=float),
            "poly_xz": poly_arr,
            "obb_corners": corners,
        })

    astar_out = []
    for path in astar_paths or []:
        astar_out.append({
            "segment": int(path.get("segment", -1)),
            "from_door": int(path.get("from_door", -1)),
            "to_door": int(path.get("to_door", -1)),
            "points": np.asarray(path.get("points", []), float),
        })

    traj_out = {
        "status": qp3d.get("status", ""),
        "obj": float(qp3d.get("obj", 0.0)),
        "sample_t": np.asarray(qp3d.get("sample_t", []), float),
        "sample_xyz": np.asarray(qp3d.get("sample_xyz", []), float),
        "sample_seg": np.asarray(qp3d.get("sample_seg", []), int),
        "T_per_seg": np.asarray(qp3d.get("T_per_seg", []), float),
        "coeffs_x": np.asarray(qp3d.get("coeffs_x", []), float),
        "coeffs_y": np.asarray(qp3d.get("coeffs_y", []), float),
        "coeffs_z": np.asarray(qp3d.get("coeffs_z", []), float),
        "order": int(qp3d.get("order", 0)),
        "hard_idx": np.asarray(qp3d.get("hard_idx", []), int),
    }

    keyframes_out = {
        "P_wp": np.asarray(P_wp, float),
        "roll_wp": np.asarray(roll_wp, float),
        "tags": np.asarray(tags, dtype=object),
        "hard": np.asarray(hard, bool),
        "door_ids": np.asarray(door_ids, int),
    }

    export = {
        "traj": traj_out,
        "keyframes": keyframes_out,
        "doors": np.asarray(doors_out, dtype=object),
        "forbidden": np.asarray(forb_out, dtype=object),
        "segments": np.asarray(seg_out, dtype=object),
        "astar_paths": np.asarray(astar_out, dtype=object),
    }

    savemat(out_path, export, do_compression=True)
    print(f"[export][matlab] saved path={out_path}")


def main():
    MODEL_PATH = "pass_planner_sac_grid"  # 你确认的名字（SB3 会自动找 .zip）:contentReference[oaicite:4]{index=4}

    # 1) 场景门洞
    doors = make_fixed_doors()
    doors = sorted(doors, key=door_y_mid)

    # 2) env + model
    env = PassEnvGrid(grid_n=16, grid_samples=3)
    model = SAC.load(MODEL_PATH, env=env)

    # 3) 逐门洞推理（theta_prev 串联：0 -> theta1 -> theta2 -> ...）
    theta_prev = 0.0
    sols = []
    for i, door in enumerate(doors):
        door_poly = door["poly"]
        cx, cz, theta, info = solve_passing_for_door_grid(env, model, door_poly, theta_prev)

        print(
            f"[door][solve] idx={i} theta_prev={theta_prev:+.3f} "
            f"theta={theta:+.3f} cx={cx:.3f} cz={cz:.3f} inside={info.get('all_inside', None)}"
        )

        sols.append({"cx": cx, "cz": cz, "theta": theta})
        theta_prev = theta  # 关键串联

    # 4) 构造 keyframes（先用插值跑通，再上 A*）
    start = np.array([0.0, doors[0]["y_min"] - 2.0, 0.5], float)
    goal  = np.array([0.0, doors[-1]["y_max"] + 2.0, 0.5], float)

    # 仅保留门与门之间的障碍，不再使用门框/墙体环带
    forbidden = []

    # 在每对相邻门之间放置一个矩形障碍，使 A* 必须绕开但仍有可行通路
    for i in range(len(doors) - 1):
        d1 = doors[i]
        d2 = doors[i + 1]
        # 中心点为两门中心的线段中点（XZ 平面）
        cx = 0.5 * (float(d1["door_cx"]) + float(d2["door_cx"]))
        cz = 0.5 * (float(d1["door_cz"]) + float(d2["door_cz"]))
        # 取 y 在两门之间（靠近中间），障碍在局部 y 范围内
        y_mid = 0.5 * (float(d1["y_max"]) + float(d2["y_min"]))

        # 居中放置，尺寸更大并膨胀，迫使路径远离障碍
        x_offset = 0.0

        size = 0.85  # 边长更大
        h = size * 0.5
        poly = Polygon([
            (cx + x_offset - h, cz - h),
            (cx + x_offset + h, cz - h),
            (cx + x_offset + h, cz + h),
            (cx + x_offset - h, cz + h),
        ]).buffer(0.15)  # 更强膨胀障碍

        forbidden.append({
            "poly_xz": poly,
            "y_min": float(y_mid - 0.55),
            "y_max": float(y_mid + 0.55),
            "tag": f"mid_obs_{i}",
        })



    P_wp, roll_wp, tags, hard, door_ids, astar_paths = build_keyframes_with_astar(
        doors, sols, start, goal, forbidden,
        pre_dist=0.35, post_dist=0.35,
        corridor_radius=4.5,
        corridor_safety=0.35,
        debug_keyframes=True,
        return_astar_paths=True,
    )
    
    # 3D corridor (AABB) from keyframes/A*
    forb_boxes = []
    y_margin = 0.2  # 避免过度收缩导致盒子被“掐没”
    for f in forbidden:
        if "poly_xz" in f:
            xmin, zmin, xmax, zmax = f["poly_xz"].bounds
            # 同步提供 poly_xz，便于 tube+Y 重叠检测
            from shapely.geometry import Polygon as _Poly
            rect_poly = _Poly([(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)])
            forb_boxes.append({
                # provide both underscore and non-underscore keys to match build_box_corridor_constraints expectations
                "xmin": float(xmin), "xmax": float(xmax),
                "ymin": float(f["y_min"]) + y_margin, "ymax": float(f["y_max"]) - y_margin,
                "zmin": float(zmin), "zmax": float(zmax),
                "x_min": float(xmin), "x_max": float(xmax),
                "y_min": float(f["y_min"]) + y_margin, "y_max": float(f["y_max"]) - y_margin,
                "z_min": float(zmin), "z_max": float(zmax),
                "poly_xz": rect_poly.buffer(0.0),
            })

    box_segments = build_box_corridor_constraints(
        P_wp=P_wp,
        hard=hard,
        tags=tags,
        door_ids=door_ids,
        forbidden_prisms=forb_boxes,
        margin_xz=0.12,
        margin_y=0.08,
        shrink=0.6,
        max_shrink_iter=35,
    )
    # =========================
    # Step4: corridor 失败段 -> 提升 hard 点并重建 corridor
    # 插入位置：build_box_corridor_constraints(...) 之后，time allocation 之前
    # =========================
    disabled = [seg for seg in box_segments if not seg.get("enabled", True)]
    if disabled:
        print(f"[corridor][promote-hard] disabled={len(disabled)} -> promote points to hard and rebuild corridors")

        # 1) 把所有 disabled 段覆盖到的 waypoint（含端点）提升为 hard
        for seg in disabled:
            i0, i1 = seg["idx_range"]          # (i0,i1) in P_wp
            hard[i0:i1 + 1] = True

        # 2) 用新的 hard 重新构 corridor（hard 变多，段会更短，corridor更容易不穿）
        box_segments = build_box_corridor_constraints(
            P_wp=P_wp,
            hard=hard,
            tags=tags,
            door_ids=door_ids,
            forbidden_prisms=forb_boxes,
            margin_xz=0.12,
            margin_y=0.08,
            shrink=0.6,
            max_shrink_iter=35,
        )

        # 3) 打印一下 hard 增量（可选）
        print(f"[corridor][promote-hard] new_hard_count={int(hard.sum())}/{hard.shape[0]}")
        print(f"[corridor][promote-hard] segments_total={len(box_segments)} enabled={sum(int(s.get('enabled', True)) for s in box_segments)}")

    # angle-aware time allocation per segment (give turns more time)
    # 保守时间分配：降低速度、增加角度权重、抬高最短时间
    v_nom = 0.9
    t_min, t_max = 0.8, 10.0
    angle_gain = 1.1
    time_scale = 1.4  # 全局放大每段时间
    hard_idx = np.where(hard)[0]
    K = hard_idx.size - 1

    seg_len = np.zeros(K, dtype=float)
    dir_xz = [None] * K
    for k, (i0, i1) in enumerate(zip(hard_idx[:-1], hard_idx[1:])):
        seg_pts = P_wp[i0:i1 + 1]
        if seg_pts.shape[0] >= 2:
            d = np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)
            seg_len[k] = float(np.sum(d))
        # heading on XZ for turn angle
        delta = P_wp[i1][[0, 2]] - P_wp[i0][[0, 2]]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            dir_xz[k] = delta / norm

    # accumulate turning angle onto adjacent segments
    ang_score = np.zeros(K, dtype=float)
    for k in range(1, K):
        u_prev = dir_xz[k - 1]
        u_curr = dir_xz[k]
        if u_prev is None or u_curr is None:
            continue
        dot = float(np.clip(np.dot(u_prev, u_curr), -1.0, 1.0))
        ang = float(np.arccos(dot))  # [0,pi]
        ang_score[k - 1] += ang
        ang_score[k] += ang

    T_list = []
    for k in range(K):
        base = seg_len[k] / max(v_nom, 1e-6)
        w = 1.0 + angle_gain * (ang_score[k] / np.pi)
        Tk = np.clip(base * w * time_scale, t_min, t_max)
        T_list.append(float(Tk))

    if len(T_list):
        print(f"[time_alloc] min/mean/max = {np.min(T_list):.3f}/{np.mean(T_list):.3f}/{np.max(T_list):.3f} (K={len(T_list)})")

    qp3d = solve_minsnap_xyz_with_box_corridors_osqp(
        P_wp=P_wp,
        hard=hard,
        segments=box_segments,
        T_per_seg=T_list,
        n_ineq_samples=12,
        eps_corridor=2e-2,
        verbose=True,
    )

    print(f"[qp3d] status={qp3d['status']} obj={qp3d['obj']}")
    np.save("qp3d_sample_t.npy", qp3d["sample_t"])
    np.save("qp3d_sample_xyz.npy", qp3d["sample_xyz"])

    print(f"\n[qp_validation] Checking 3D trajectory against box corridors...")
    max_v, bad, tot, rate = check_traj_in_box_corridor(
        qp3d["sample_xyz"], box_segments, qp3d.get("sample_seg"), tol=1e-6
    )
    print(f"[qp_validation][box] max_violation={max_v:.6f} bad_samples={bad}/{tot} rate={rate:.4f}")
    if rate > 0.01:
        print(f"[qp_validation][WARNING] {rate*100:.2f}% of samples outside corridor")
    else:
        print(f"[qp_validation][OK] trajectory satisfies corridors (rate<1%)")

    # 额外：真实障碍碰撞校验（基于 poly_xz + y 区间）
    def validate_traj_vs_forbidden(sample_xyz: np.ndarray, forbidden_prisms: list, step: int = 1):
        from shapely.geometry import LineString as _Line
        M = sample_xyz.shape[0]
        hit_cnt = 0
        for i in range(0, M - 1, max(1, step)):
            a = sample_xyz[i]
            b = sample_xyz[min(i + 1, M - 1)]
            y0, y1 = float(a[1]), float(b[1])
            lz = _Line([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])
            for f in forbidden:
                fy0, fy1 = float(f["y_min"]), float(f["y_max"])
                if max(y0, y1) < fy0 or min(y0, y1) > fy1:
                    continue
                if lz.intersects(f["poly_xz"]):
                    hit_cnt += 1
                    break
        return hit_cnt

    hit = validate_traj_vs_forbidden(qp3d["sample_xyz"], forbidden, step=1)
    if hit > 0:
        print(f"[qp_validation][forbidden][ALERT] segment hits={hit} -> 轨迹与真实障碍发生相交！")
    else:
        print(f"[qp_validation][forbidden][OK] 轨迹未与任何真实障碍相交（无穿模）。")

    print(f"\n[viz] Generating 3D corridor projections...")
    visualize_box_corridor_and_trajectory(
        segments=box_segments,
        sample_xyz=qp3d["sample_xyz"],
        P_wp=P_wp,
        forbidden_prisms=forbidden,
        out_path_xz="corridor_traj_xz.png",
        out_path_xy="corridor_traj_xy.png",
        out_path_yz="corridor_traj_yz.png",
        out_path_svg="corridor_traj_all.svg",
        show=SHOW_PLOTS,
    )
    
    # 增强可视化：标记硬软约束点
    print(f"\n[viz] Generating keyframes detail plot...")
    visualize_keyframes_detail(P_wp, hard, tags, door_ids, "keyframes_detail_xz.png", plane="xz")
    visualize_keyframes_detail(P_wp, hard, tags, door_ids, "keyframes_detail_xy.png", plane="xy")
    visualize_keyframes_detail(P_wp, hard, tags, door_ids, "keyframes_detail_yz.png", plane="yz")

    print(f"\n[corridor3d] segments_total={len(box_segments)} enabled={sum(int(s.get('enabled', True)) for s in box_segments)}")
    print(f"[summary][keyframes] P_shape={P_wp.shape} roll_shape={roll_wp.shape}")
    print(f"[summary][tags] unique_counts={np.unique(tags, return_counts=True)}")
    print(f"[summary][hard] count={int(hard.sum())}/{hard.shape[0]} indices_first40={np.where(hard)[0][:40]}")
    print(f"[summary][door_ids] unique_counts={np.unique(door_ids, return_counts=True)}")
    print(f"[summary][P_wp] shape={P_wp.shape}")
    print(f"[summary][P_wp_first8]\n{np.array2string(P_wp[:8], precision=3, suppress_small=True)}")
    np.save("debug_P_wp.npy", P_wp)

    visualize_passing_xz(doors, sols, env, out_path="passing_xz_debug.png")
    visualize_astar_debug_xz(doors, forbidden, P_wp, out_path="astar_debug_xz.png")
    debug_segment_intersections(P_wp, forbidden, seg_name="(keyframes)")
    visualize_astar_debug_xy(doors, forbidden, P_wp, out_path="astar_debug_xy.png")

    export_to_matlab(
        out_path="corridor_export.mat",
        qp3d=qp3d,
        P_wp=P_wp,
        roll_wp=roll_wp,
        tags=tags,
        hard=hard,
        door_ids=door_ids,
        doors=doors,
        forbidden=forbidden,
        box_segments=box_segments,
        astar_paths=astar_paths,
    )


    # 5) TODO：把 P_wp 喂给你旧版的 min-snap 模块（这里保持旧版实现不动）
    #   - 如果旧版 fixed_scene_3d_demo.py 里有 solve_min_snap(P_wp, roll_wp, ...)
    #     直接 import 过来调用即可。

if __name__ == "__main__":
    main()
