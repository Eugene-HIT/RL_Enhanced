import numpy as np
import matplotlib.pyplot as plt

from env_passenv_grid import PassEnvGrid

import scipy.sparse as sp

def _poly_basis(t: float, order: int = 7, deriv: int = 0) -> np.ndarray:
    """Return basis vector for d^deriv/dt^deriv of [1, t, t^2, ..., t^order]."""
    v = np.zeros(order + 1, dtype=float)
    for i in range(order + 1):
        if i < deriv:
            continue
        c = 1.0
        # i*(i-1)*...*(i-deriv+1)
        for k in range(deriv):
            c *= (i - k)
        v[i] = c * (t ** (i - deriv))
    return v

def _snap_Q(T: float, order: int = 7) -> np.ndarray:
    """
    Q such that integral_0^T (p''''(t))^2 dt = a^T Q a,
    where p(t)=sum_i a_i t^i.
    """
    n = order + 1
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i < 4:
            continue
        ci = i * (i - 1) * (i - 2) * (i - 3)
        for j in range(n):
            if j < 4:
                continue
            cj = j * (j - 1) * (j - 2) * (j - 3)
            p = (i - 4) + (j - 4)  # power of t in product
            # ∫0^T t^p dt = T^(p+1)/(p+1)
            Q[i, j] = ci * cj * (T ** (p + 1)) / (p + 1)
    return Q

def _time_allocation_from_segments(segments, v_nom=1.0, t_min=0.25, t_max=8.0):
    """
    Allocate duration per segment based on XZ polyline length (from P_seg).
    v_nom: nominal speed in "meters/sec" scale of your coordinates.
    """
    Ts = []
    for seg in segments:
        P = seg["P_seg"]
        xz = P[:, [0, 2]]
        length = float(np.sum(np.linalg.norm(np.diff(xz, axis=0), axis=1)))
        T = max(t_min, min(t_max, length / max(v_nom, 1e-6)))
        Ts.append(T)
    return np.asarray(Ts, dtype=float)

def solve_minsnap_xz_with_corridors_osqp(
    segments,
    P_wp: np.ndarray,
    hard: np.ndarray,
    order: int = 7,
    v_nom: float = 1.0,
    n_sample_ineq: int = 6,
    w_snap: float = 1.0,
    w_reg: float = 1e-8,
    debug: bool = False,
):
    """
    Solve a joint QP for x(t), z(t) with corridor halfspace constraints.

    segments: output of build_corridor_constraints(...)
      each seg contains:
        - i0, i1 (hard indices in P_wp)
        - A_xz, b_xz (halfspace on [x,z])
        - P_seg (polyline points used to build corridor)
    P_wp: (N,3) keyframes (hard+soft)
    hard: (N,) bool, hard anchors (start, door_pre/mid/post, goal)

    Return dict:
      coeffs_x: (K,8), coeffs_z: (K,8), T: (K,)
      sample_t: (M,), sample_xz: (M,2), seg_id_of_sample: (M,)
    """
    import osqp

    K = len(segments)
    n = order + 1         # 8
    nx_seg = n            # per dim
    # decision per segment: [a_x(8), a_z(8)] => 16
    dim_seg = 2 * n
    dim = K * dim_seg

    # --- durations ---
    T = _time_allocation_from_segments(segments, v_nom=v_nom)

    # --- Quadratic cost P ---
    # block diag of (Qx, Qz) for each segment
    P_blocks = []
    for k in range(K):
        Q = _snap_Q(float(T[k]), order=order)
        Q = w_snap * Q + w_reg * np.eye(n)
        Pk = sp.block_diag((Q, Q), format="csc")  # (16x16)
        P_blocks.append(Pk)
    Pmat = sp.block_diag(P_blocks, format="csc")

    q = np.zeros(dim, dtype=float)

    # Helpers: index mapping
    def idx_x(k):  # slice for x coeffs of segment k
        s = k * dim_seg
        return slice(s, s + n)

    def idx_z(k):
        s = k * dim_seg + n
        return slice(s, s + 2 * n)

    # --- Equality constraints ---
    Aeq_rows = []
    beq = []

    def add_eq_row(row_dict, rhs):
        # row_dict: {col_index: value}
        Aeq_rows.append(row_dict)
        beq.append(float(rhs))

    # 1) Endpoint position constraints for each segment
    for k, seg in enumerate(segments):
        i0 = seg["i0"]
        i1 = seg["i1"]
        p0 = P_wp[i0]
        p1 = P_wp[i1]
        Tk = float(T[k])

        phi0 = _poly_basis(0.0, order=order, deriv=0)
        phiT = _poly_basis(Tk,  order=order, deriv=0)

        # x(0)=p0.x ; z(0)=p0.z ; x(T)=p1.x ; z(T)=p1.z
        # x(0)
        row = {}
        base = k * dim_seg
        for j in range(n):
            row[base + j] = phi0[j]
        add_eq_row(row, p0[0])
        # z(0)
        row = {}
        base = k * dim_seg + n
        for j in range(n):
            row[base + j] = phi0[j]
        add_eq_row(row, p0[2])

        # x(T)
        row = {}
        base = k * dim_seg
        for j in range(n):
            row[base + j] = phiT[j]
        add_eq_row(row, p1[0])
        # z(T)
        row = {}
        base = k * dim_seg + n
        for j in range(n):
            row[base + j] = phiT[j]
        add_eq_row(row, p1[2])

    # 2) C3 continuity between segments (v,a,j) at junctions
    for k in range(K - 1):
        Tk = float(T[k])
        phi_vT = _poly_basis(Tk, order=order, deriv=1)
        phi_aT = _poly_basis(Tk, order=order, deriv=2)
        phi_jT = _poly_basis(Tk, order=order, deriv=3)

        phi_v0 = _poly_basis(0.0, order=order, deriv=1)
        phi_a0 = _poly_basis(0.0, order=order, deriv=2)
        phi_j0 = _poly_basis(0.0, order=order, deriv=3)

        for deriv_phiT, deriv_phi0 in [(phi_vT, phi_v0), (phi_aT, phi_a0), (phi_jT, phi_j0)]:
            # x continuity: x_k^{(d)}(T) - x_{k+1}^{(d)}(0) = 0
            row = {}
            base_k = k * dim_seg
            base_n = (k + 1) * dim_seg
            for j in range(n):
                row[base_k + j] = deriv_phiT[j]
                row[base_n + j] = row.get(base_n + j, 0.0) - deriv_phi0[j]
            add_eq_row(row, 0.0)

            # z continuity
            row = {}
            base_k = k * dim_seg + n
            base_n = (k + 1) * dim_seg + n
            for j in range(n):
                row[base_k + j] = deriv_phiT[j]
                row[base_n + j] = row.get(base_n + j, 0.0) - deriv_phi0[j]
            add_eq_row(row, 0.0)

    # 3) Boundary derivatives at start/end: v=a=j=0
    # start segment k=0 at t=0
    for d in [1, 2, 3]:
        phi = _poly_basis(0.0, order=order, deriv=d)
        # x
        row = {}
        base = 0 * dim_seg
        for j in range(n):
            row[base + j] = phi[j]
        add_eq_row(row, 0.0)
        # z
        row = {}
        base = 0 * dim_seg + n
        for j in range(n):
            row[base + j] = phi[j]
        add_eq_row(row, 0.0)

    # end segment k=K-1 at t=T
    Tk = float(T[-1])
    for d in [1, 2, 3]:
        phi = _poly_basis(Tk, order=order, deriv=d)
        # x
        row = {}
        base = (K - 1) * dim_seg
        for j in range(n):
            row[base + j] = phi[j]
        add_eq_row(row, 0.0)
        # z
        row = {}
        base = (K - 1) * dim_seg + n
        for j in range(n):
            row[base + j] = phi[j]
        add_eq_row(row, 0.0)

    # Build sparse Aeq
    n_eq = len(beq)
    data = []
    rows = []
    cols = []
    for r, rowdict in enumerate(Aeq_rows):
        for c, v in rowdict.items():
            rows.append(r)
            cols.append(c)
            data.append(v)
    Aeq = sp.csc_matrix((data, (rows, cols)), shape=(n_eq, dim))
    leq = np.asarray(beq, dtype=float)
    ueq = np.asarray(beq, dtype=float)

    # --- Inequality constraints: corridor halfspaces sampled along each segment ---
    Ainq_list = []
    uinq_list = []

    # We won’t set lower bound (=-inf), only upper bound
    # For each sample: A_xz @ [x(t), z(t)] <= b_xz
    for k, seg in enumerate(segments):
        A_xz = seg.get("A_xz", None)
        b_xz = seg.get("b_xz", None)
        if A_xz is None or b_xz is None:
            continue
        A_xz = np.asarray(A_xz, float)
        b_xz = np.asarray(b_xz, float)
        if A_xz.ndim != 2 or A_xz.shape[1] != 2:
            continue

        Tk = float(T[k])

        # sample times in (0,T), include a bit away from endpoints
        if n_sample_ineq <= 1:
            ts = np.array([0.5 * Tk], dtype=float)
        else:
            ts = np.linspace(0.05 * Tk, 0.95 * Tk, n_sample_ineq)

        for t in ts:
            phi = _poly_basis(float(t), order=order, deriv=0)  # (8,)
            # each halfspace row => one inequality
            for r_h in range(A_xz.shape[0]):
                ax, az = float(A_xz[r_h, 0]), float(A_xz[r_h, 1])
                row = {}
                base = k * dim_seg
                # x part
                for j in range(n):
                    row[base + j] = ax * phi[j]
                # z part
                basez = k * dim_seg + n
                for j in range(n):
                    row[basez + j] = row.get(basez + j, 0.0) + az * phi[j]

                # store
                Ainq_list.append(row)
                uinq_list.append(float(b_xz[r_h]))

    # Build sparse Ainq
    n_inq = len(uinq_list)
    if n_inq > 0:
        data = []
        rows = []
        cols = []
        for r, rowdict in enumerate(Ainq_list):
            for c, v in rowdict.items():
                rows.append(r)
                cols.append(c)
                data.append(v)
        Ainq = sp.csc_matrix((data, (rows, cols)), shape=(n_inq, dim))
        linq = -np.inf * np.ones(n_inq, dtype=float)
        uinq = np.asarray(uinq_list, dtype=float)
        Aall = sp.vstack([Aeq, Ainq], format="csc")
        lall = np.concatenate([leq, linq])
        uall = np.concatenate([ueq, uinq])
    else:
        Aall = Aeq
        lall = leq
        uall = ueq

    if debug:
        print("[QP] dim:", dim, "K:", K, "eq:", n_eq, "inq:", n_inq)

    # --- Solve with OSQP ---
    prob = osqp.OSQP()
    prob.setup(P=Pmat, q=q, A=Aall, l=lall, u=uall,
               verbose=debug, polish=True, eps_abs=1e-5, eps_rel=1e-5,
               max_iter=20000)
    res = prob.solve()

    if res.info.status_val not in (1, 2):  # 1=solved, 2=solved inaccurate
        raise RuntimeError(f"OSQP failed: {res.info.status} (val={res.info.status_val})")

    x = res.x  # decision vector

    coeffs_x = np.zeros((K, n), dtype=float)
    coeffs_z = np.zeros((K, n), dtype=float)
    for k in range(K):
        coeffs_x[k] = x[idx_x(k)]
        coeffs_z[k] = x[idx_z(k)]

    # --- Sample trajectory (xz only) ---
    # uniform sampling per segment using dt based on each segment duration
    sample_t = []
    sample_xz = []
    seg_ids = []
    t_global = 0.0
    for k in range(K):
        Tk = float(T[k])
        m = max(2, int(np.ceil(Tk / 0.08)))  # default sampling step ~0.08
        ts = np.linspace(0.0, Tk, m)
        ax = coeffs_x[k]
        az = coeffs_z[k]
        for tt in ts:
            phi = _poly_basis(float(tt), order=order, deriv=0)
            xx = float(phi @ ax)
            zz = float(phi @ az)
            sample_t.append(t_global + float(tt))
            sample_xz.append([xx, zz])
            seg_ids.append(k)
        t_global += Tk

    return {
        "coeffs_x": coeffs_x,
        "coeffs_z": coeffs_z,
        "T": T,
        "sample_t": np.asarray(sample_t, float),
        "sample_xz": np.asarray(sample_xz, float),
        "sample_seg": np.asarray(seg_ids, int),
        "osqp_status": res.info.status,
        "osqp_obj": float(res.info.obj_val),
    }


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
