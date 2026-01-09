import numpy as np
import matplotlib.pyplot as plt

from env_passenv_grid import PassEnvGrid


def plot_polygon(ax, poly, title=""):
    xy = np.array(poly.exterior.coords)
    ax.plot(xy[:, 0], xy[:, 1], "k-", lw=2)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)

    xmin, ymin, xmax, ymax = poly.bounds
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            linestyle="--", color="0.5", lw=1)


def plot_grid(ax, grid):
    im = ax.imshow(grid, origin="lower")
    ax.set_title("Grid observation (p_free)")
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    env = PassEnvGrid(grid_n=16, grid_samples=3)

    N = 12   # 看多少个 reset 样本
    for i in range(N):
        obs, _ = env.reset()

        poly = env.door_poly
        grid = env._grid_cached
        bbox = env.door_bbox
        shape_type = env.shape_type

        xmin, zmin, xmax, zmax = bbox
        w = xmax - xmin
        h = zmax - zmin

        print(f"\n[{i}] shape_type = {shape_type}")
        print(f"    bbox size = ({w:.2f}, {h:.2f})")
        print(f"    theta_prev(deg) = {np.rad2deg(env.theta_prev):.1f}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        plot_polygon(
            axs[0],
            poly,
            title=f"Door geometry (shape_type={shape_type})"
        )

        plot_grid(axs[1], grid)

        plt.tight_layout()
        plt.show()

    env.close()
