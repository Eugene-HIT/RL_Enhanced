# test_ao_shapes_grid.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from stable_baselines3 import SAC
from env_passenv_grid import PassEnvGrid
from geom_encoder import polygon_to_grid


def make_ao_polygon(kind="up",   # "up"=凹（上凹），"down"=倒凹（下凹）
                    center=(1.2, 1.0),
                    outer_w=1.2,
                    outer_h=1.8,
                    notch_w=0.45,
                    notch_d=0.55):
    """
    生成“凹 / 倒凹”门洞轮廓：
    - 先构造一个外框矩形
    - 在顶部或底部挖一个矩形凹槽（凹进去的区域为不可通行）
    返回的是门洞的边界多边形（凹形轮廓），适合 within 判定。
    """
    cx, cz = center
    W, H = float(outer_w), float(outer_h)
    nw, nd = float(notch_w), float(notch_d)

    xmin, xmax = cx - W/2, cx + W/2
    zmin, zmax = cz - H/2, cz + H/2

    # notch x-range centered
    nx0, nx1 = cx - nw/2, cx + nw/2

    if kind == "up":
        # 上凹：从顶部往下挖 nd 深度
        # notch bottom z
        nz = zmax - nd

        # 多边形按逆时针绕一圈，顶部中间凹进去
        pts = np.array([
            [xmin, zmin],
            [xmax, zmin],
            [xmax, zmax],
            [nx1, zmax],
            [nx1, nz],
            [nx0, nz],
            [nx0, zmax],
            [xmin, zmax],
        ], dtype=float)

    elif kind == "down":
        # 下凹：从底部往上挖 nd 深度
        nz = zmin + nd
        pts = np.array([
            [xmin, zmin],
            [nx0, zmin],
            [nx0, nz],
            [nx1, nz],
            [nx1, zmin],
            [xmax, zmin],
            [xmax, zmax],
            [xmin, zmax],
        ], dtype=float)
    else:
        raise ValueError("kind must be 'up' or 'down'")

    return Polygon(pts).buffer(0.0)


def box_corners(cx, cz, th, L, H):
    base = np.array([
        [-L/2, -H/2],
        [ L/2, -H/2],
        [ L/2,  H/2],
        [-L/2,  H/2],
    ], dtype=float)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=float)
    rot = (R @ base.T).T
    return rot + np.array([cx, cz], dtype=float)


def run_one(model, env, door_poly, title="test"):
    env.door_poly = door_poly
    grid, center, bbox, _ = polygon_to_grid(door_poly, n=env.grid_n, samples=env.grid_samples, return_meta=True)
    env._grid_cached = grid.astype(np.float32)
    env.door_center = center.astype(np.float32)
    env.door_bbox = bbox
    env.shape_type = 888

    env.theta_prev = float(env.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))
    obs = env._build_obs(env._grid_cached, env.door_center, env.theta_prev)

    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)

    print(f"\n===== {title} =====")
    print("all_inside:", info["all_inside"])
    print("reward:", reward)
    print("min_margin:", info["min_margin"])
    print("theta_prev(deg):", np.rad2deg(info["theta_prev"]))
    print("theta(deg):     ", np.rad2deg(info["theta"]))
    print("cx, cz:", info["cx"], info["cz"])
    print("action:", action)

    door_xy = np.array(door_poly.exterior.coords, dtype=float)
    corners = box_corners(info["cx"], info["cz"], info["theta"], env.L_eff, env.H_eff)
    corners = np.vstack([corners, corners[0]])

    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title)
    plt.plot(door_xy[:, 0], door_xy[:, 1], "k-", lw=2, label="door")
    plt.plot(corners[:, 0], corners[:, 1], "b-", lw=2, label="payload")
    plt.scatter([info["cx"]], [info["cz"]], c="b", s=40)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "pass_planner_sac_grid"
    env = PassEnvGrid(grid_n=16, grid_samples=3)
    model = SAC.load(MODEL_PATH, env=env)

    center = (1.2, 1.0)

    # 你可以调 notch_w / notch_d 来制造更难的“凹槽”
    poly_ao  = make_ao_polygon(kind="up",   outer_w=1.2, outer_h=1.8, notch_w=0.95, notch_d=1.05, center=center)
    poly_dao = make_ao_polygon(kind="down", outer_w=1.2, outer_h=1.8, notch_w=0.95, notch_d=1.05, center=center)



    run_one(model, env, poly_ao, title="凹字形（上凹 notch）")
    run_one(model, env, poly_dao, title="倒凹（下凹 notch）")
