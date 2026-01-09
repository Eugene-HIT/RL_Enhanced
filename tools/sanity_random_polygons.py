# sanity_random_polygons.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from tools.geom_random import sample_random_door_polygon
from geom_encoder import polygon_to_grid

def plot_polygon(ax, poly: Polygon):
    xy = np.array(poly.exterior.coords, dtype=float)
    ax.plot(xy[:, 0], xy[:, 1], "k-", lw=2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    xmin, zmin, xmax, zmax = poly.bounds
    # bbox
    ax.plot([xmin, xmax, xmax, xmin, xmin],
        [zmin, zmin, zmax, zmax, zmin],
        linestyle="--", linewidth=1, color="0.5")


    # bbox center
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    ax.scatter([cx], [cz], c="r", s=30)


def plot_grid(ax, grid, title="grid (p_free)"):
    # grid[iz, ix] : iz is z direction
    im = ax.imshow(grid, origin="lower")  # origin="lower" makes z increase upwards
    ax.set_aspect('equal', adjustable='box')


if __name__ == "__main__":
    # ----- config -----
    N = 10                 # 生成多少个样本看看，设置为10以匹配两个窗口各5个
    grid_n = 16
    grid_samples = 3

    # 生成参数（你可以改这些做“更随机/更凹”）
    center_base = (1.2, 1.0)
    concave_prob = 0.8
    n_vertices_range = (10, 18)
    n_notches_range = (0, 3)
    scale_x_range = (0.9, 1.8)
    scale_z_range = (0.9, 1.8)

    # ----- generate + plot -----
    # 先收集所有poly和grid
    polys = []
    grids = []
    metas = []
    all_xmin, all_xmax, all_zmin, all_zmax = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    for i in range(N):
        # 给每个样本一个 seed，方便复现
        seed = 1000 + i

        # 给中心一点点随机扰动（贴近你 env 的分布）
        cz_jitter = np.random.uniform(-0.1, 0.1)
        center = (center_base[0], center_base[1] + cz_jitter)

        poly, verts = sample_random_door_polygon(
            seed=seed,
            center=center,
            concave_prob=concave_prob,
            n_vertices_range=n_vertices_range,
            n_notches_range=n_notches_range,
            scale_x_range=scale_x_range,
            scale_z_range=scale_z_range,
        )

        # 基本检查
        xmin, zmin, xmax, zmax = poly.bounds
        print(f"\n[{i}] seed={seed}")
        print("  valid:", poly.is_valid, "empty:", poly.is_empty)
        print("  area :", float(poly.area))
        print("  bounds:", (xmin, zmin, xmax, zmax))
        print("  n_verts:", len(verts))

        # 更新全局bounds
        all_xmin = min(all_xmin, xmin)
        all_xmax = max(all_xmax, xmax)
        all_zmin = min(all_zmin, zmin)
        all_zmax = max(all_zmax, zmax)

        # encode grid
        grid, center_out, bbox, meta = polygon_to_grid(
            poly, n=grid_n, samples=grid_samples, return_meta=True
        )

        polys.append(poly)
        grids.append(grid)
        metas.append(meta)

    # 现在画图
    # 第一个窗口：样本 0-4
    fig1 = plt.figure(figsize=(15, 8), constrained_layout=True)
    for i in range(5):
        poly = polys[i]
        grid = grids[i]
        meta = metas[i]

        # 画轮廓 - 第一行
        ax1 = fig1.add_subplot(2, 5, i + 1)
        plot_polygon(ax1, poly)
        ax1.set_xlim(all_xmin, all_xmax)
        ax1.set_ylim(all_zmin, all_zmax)

        # 画 grid - 第二行
        ax2 = fig1.add_subplot(2, 5, i + 6)
        plot_grid(ax2, grid)

    plt.show()

    # 第二个窗口：样本 5-9
    fig2 = plt.figure(figsize=(15, 8), constrained_layout=True)
    for i in range(5, 10):
        j = i - 5
        poly = polys[i]
        grid = grids[i]
        meta = metas[i]

        # 画轮廓 - 第一行
        ax1 = fig2.add_subplot(2, 5, j + 1)
        plot_polygon(ax1, poly)
        ax1.set_xlim(all_xmin, all_xmax)
        ax1.set_ylim(all_zmin, all_zmax)

        # 画 grid - 第二行
        ax2 = fig2.add_subplot(2, 5, j + 6)
        plot_grid(ax2, grid)

    plt.show()
