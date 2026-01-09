# test_flipped_C_grid.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from stable_baselines3 import SAC
from env_passenv_grid import PassEnvGrid

def make_flipped_c_polygon(scale_x=1.20, scale_z=1.70, center_x=1.2, center_z=1.0):
    """
    倒C形：把原C形的 x 镜像 (开口朝左)
    返回 world 坐标下的 Polygon
    """
    pts = np.array([
        [-0.4, -0.6],
        [ 0.4, -0.6],
        [ 0.4, -0.3],
        [ 0.0, -0.3],
        [ 0.0,  0.3],
        [ 0.4,  0.3],
        [ 0.4,  0.6],
        [-0.4,  0.6],
    ], dtype=float)

    # x 镜像：开口朝左
    pts[:, 0] *= -1.0
    pts -= pts.mean(axis=0)

    pts_world = np.column_stack([
        scale_x * pts[:, 0] + center_x,
        scale_z * pts[:, 1] + center_z
    ])
    return Polygon(pts_world).buffer(0.0)


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


if __name__ == "__main__":
    MODEL_PATH = "pass_planner_sac_grid"  # 会自动找 .zip
    env = PassEnvGrid(grid_n=16, grid_samples=3)

    # 加载模型
    model = SAC.load(MODEL_PATH, env=env)

    # ============ 手动构造“倒C门洞”，并注入 env ============
    door_poly = make_flipped_c_polygon(scale_x=1.20, scale_z=1.70, center_x=1.2, center_z=1.0)
    env.door_poly = door_poly

    # 用 encoder 得到 grid/center/bbox，并缓存到 env（与 reset 保持一致）
    from geom_encoder import polygon_to_grid
    grid, center, bbox, _ = polygon_to_grid(door_poly, n=env.grid_n, samples=env.grid_samples, return_meta=True)

    env._grid_cached = grid.astype(np.float32)
    env.door_center = center.astype(np.float32)
    env.door_bbox = bbox
    env.shape_type = 3  # C形，只是 info 里好看，无所谓

    # 随机一个初始姿态（theta_prev），并且喂进 obs（你训练时就是这样）
    env.theta_prev = float(env.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))

    # 构造 obs（和 env.reset 输出一致）
    obs = env._build_obs(env._grid_cached, env.door_center, env.theta_prev)

    # ============ 推理并 step ============
    action, _ = model.predict(obs, deterministic=True)
    obs2, reward, terminated, truncated, info = env.step(action)

    # ============ 打印结果 ============
    print("\n====== Flipped C (opening LEFT) Test ======")
    print("action [ax, az, atheta]:", action)
    print("theta_prev (deg):", np.rad2deg(info["theta_prev"]))
    print("theta (deg):     ", np.rad2deg(info["theta"]))
    print("cx, cz:          ", info["cx"], info["cz"])
    print("all_inside:      ", info["all_inside"])
    print("min_margin:      ", info["min_margin"])
    print("reward:          ", reward)

    # ============ 画图 ============
    door_xy = np.array(door_poly.exterior.coords, dtype=float)

    corners = box_corners(info["cx"], info["cz"], info["theta"], env.L_eff, env.H_eff)
    corners = np.vstack([corners, corners[0]])

    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Flipped C-shape Test (SAC-grid)")

    plt.plot(door_xy[:, 0], door_xy[:, 1], "k-", lw=2, label="door")
    plt.plot(corners[:, 0], corners[:, 1], "b-", lw=2, label="payload")
    plt.scatter([info["cx"]], [info["cz"]], c="b", s=40)

    plt.legend()
    plt.show()
