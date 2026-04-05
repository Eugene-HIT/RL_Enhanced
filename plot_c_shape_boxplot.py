import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from stable_baselines3 import SAC

from env_passenv_grid import PassEnvGrid
from geom_encoder import polygon_to_grid


def _template_polygon(shape_type: int) -> np.ndarray:
    if shape_type == 0:  # rectangle
        pts = np.array([[-0.25, -0.5],
                        [ 0.25, -0.5],
                        [ 0.25,  0.5],
                        [-0.25,  0.5]], dtype=float)
    elif shape_type == 1:  # right triangle
        pts = np.array([[-0.3, -0.5],
                        [ 0.3, -0.5],
                        [-0.3,  0.5]], dtype=float)
    elif shape_type == 2:  # irregular pentagon
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


def make_fixed_doors():
    specs = [
        (3, 1.20, 1.70, (0.90, 1.00), (0.00, 2.00)),
        (2, 1.60, 1.50, (1.50, 1.10), (5.00, 7.00)),
        (1, 1.20, 1.55, (1.25, 0.95), (10.00, 12.00)),
        (0, 1.50, 1.80, (1.20, 1.05), (15.00, 17.00)),
    ]

    doors = []
    for idx, (stype, sx, sz, (cx0, cz0), (y0, y1)) in enumerate(specs, start=1):
        base = _template_polygon(stype)
        pts = np.column_stack([sx * base[:, 0], sz * base[:, 1]]) + np.array([cx0, cz0], dtype=float)
        poly = Polygon(pts).buffer(0.0)
        doors.append({
            "name": f"door{idx}",
            "shape_type": int(stype),
            "poly": poly,
            "y_min": float(y0),
            "y_max": float(y1),
        })

    return doors


def _build_obs_for_door(env: PassEnvGrid, door_poly: Polygon, theta_prev: float):
    env.door_poly = door_poly
    grid, center, bbox, _ = polygon_to_grid(
        door_poly, n=env.grid_n, samples=env.grid_samples, return_meta=True
    )
    env._grid_cached = grid.astype(np.float32)
    env.door_center = center.astype(np.float32)
    env.door_bbox = bbox
    env.theta_prev = float(theta_prev)
    return env._build_obs(env._grid_cached, env.door_center, env.theta_prev)


def run_trials(env: PassEnvGrid, model: SAC, door: dict, n_trials: int = 100, seed: int | None = None):
    rng = np.random.default_rng(seed)

    thetas = []
    margins = []
    xs = []
    zs = []

    success = 0
    for _ in range(n_trials):
        theta_prev = np.deg2rad(rng.uniform(-30.0, 30.0))
        obs = _build_obs_for_door(env, door["poly"], theta_prev)
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)

        if info.get("all_inside", False):
            success += 1
            thetas.append(np.degrees(info["theta"]))
            margins.append(info["min_margin"])
            xs.append(info["cx"])
            zs.append(info["cz"])

    stats = {
        "shape_type": int(door.get("shape_type", -1)),
        "theta_deg": np.asarray(thetas, float),
        "min_margin": np.asarray(margins, float),
        "cx": np.asarray(xs, float),
        "cz": np.asarray(zs, float),
        "success": success,
        "total": n_trials,
    }
    return stats


def plot_boxplots(stats_list):
    if not stats_list:
        raise RuntimeError("No stats to plot.")

    labels = [st.get("label", f"T{st['shape_type']}") for st in stats_list]

    def _data(key):
        return [st[key] if st["success"] > 0 else np.array([np.nan]) for st in stats_list]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].boxplot(_data("theta_deg"), labels=labels)
    axes[0, 0].set_ylabel(r"$\mathrm{Angle}\,(deg)$")

    axes[0, 1].boxplot(_data("min_margin"), labels=labels)
    axes[0, 1].set_ylabel(r"$\mathrm{Safety\ margin}\,(m)$")

    axes[1, 0].boxplot(_data("cx"), labels=labels)
    axes[1, 0].set_ylabel(r"$\mathrm{Passing}\ x\,(m)$")

    axes[1, 1].boxplot(_data("cz"), labels=labels)
    axes[1, 1].set_ylabel(r"$\mathrm{Passing}\ z\,(m)$")

    fig.suptitle("Door types T0-T3: pass statistics (successful only)")
    fig.tight_layout()
    plt.show()


def _payload_rect(cx: float, cz: float, theta: float, L: float, H: float):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    rect = np.array([
        [-0.5 * L, -0.5 * H],
        [ 0.5 * L, -0.5 * H],
        [ 0.5 * L,  0.5 * H],
        [-0.5 * L,  0.5 * H],
        [-0.5 * L, -0.5 * H],
    ], dtype=float)
    rect = rect @ R.T + np.array([cx, cz], dtype=float)
    return rect


def plot_door_examples(env: PassEnvGrid, model: SAC, doors: list[dict]):
    samples = []
    theta_prev = 0.0
    for door in doors:
        obs = _build_obs_for_door(env, door["poly"], theta_prev)
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)

        door_xy = np.asarray(info["door_xy"], float)
        cx = float(info["cx"])
        cz = float(info["cz"])
        theta = float(info["theta"])
        theta_prev = theta

        rect = _payload_rect(cx, cz, theta, env.L_eff, env.H_eff)
        samples.append({
            "door": door_xy,
            "rect": rect,
        })

    # global bounds for equal-sized subplots
    all_pts = np.vstack([s["door"] for s in samples])
    xmin, zmin = all_pts.min(axis=0)
    xmax, zmax = all_pts.max(axis=0)
    pad = 0.25
    xlim = (xmin - pad, xmax + pad)
    zlim = (zmin - pad, zmax + pad)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.reshape(-1)

    for ax, s in zip(axes, samples):
        ax.plot(s["door"][:, 0], s["door"][:, 1], linewidth=2)
        ax.plot(s["rect"][:, 0], s["rect"][:, 1], linewidth=2)
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xlabel(r"$x\,(m)$")
        ax.set_ylabel(r"$z\,(m)$")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "pass_planner_sac_grid"  # SB3 自动加载 .zip
    env = PassEnvGrid(grid_n=16, grid_samples=3)
    model = SAC.load(MODEL_PATH, env=env)

    doors = make_fixed_doors()

    stats_all = []
    for idx, door in enumerate(doors):
        stats = run_trials(env, model, door, n_trials=100, seed=None)
        stats["label"] = f"T{idx}"
        print(f"[summary][T{idx}] success={stats['success']}/{stats['total']}")
        stats_all.append(stats)

    plot_boxplots(stats_all)
    plot_door_examples(env, model, doors)
