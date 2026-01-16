import numpy as np
from shapely.geometry import LineString

import scipy.sparse as sp

from shapely.ops import unary_union

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

def convex_polygon_to_halfspace(poly: Polygon, eps: float = 1e-12):
    """
    Convert a convex Polygon (CCW) into halfspace A x <= b, where x=[x,z]^T.
    Returns:
      A: (m,2), b: (m,)
    """
    if poly.is_empty:
        return None, None
    poly = orient(poly, sign=1.0)  # CCW
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return None, None
    coords = coords[:-1]  # drop repeated last

    A = []
    b = []
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        ex = x2 - x1
        ey = y2 - y1
        # For CCW polygon, interior is to the LEFT of each directed edge.
        # Inequality in <= form can be: n·p <= n·p1, with n = [ey, -ex]
        nx, ny = ey, -ex
        norm = (nx * nx + ny * ny) ** 0.5
        if norm < eps:
            continue
        nx /= norm
        ny /= norm
        A.append([nx, ny])
        b.append(nx * x1 + ny * y1)

    if len(A) == 0:
        return None, None
    A = np.asarray(A, float)
    b = np.asarray(b, float)

    # sanity: no NaN/Inf
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        return None, None
    return A, b


def is_door_internal_segment(i0, i1, tags, door_ids):
    # 只放行 door_pre->door_mid 和 door_mid->door_post（同一扇门）
    t0, t1 = tags[i0], tags[i1]
    d0, d1 = int(door_ids[i0]), int(door_ids[i1])
    if d0 >= 0 and d0 == d1:
        ok_pairs = {("door_pre","door_mid"), ("door_mid","door_post")}
        return (t0, t1) in ok_pairs
    return False


def check_traj_in_corridor_xz(sample_xz, segments, sample_seg=None, tol=1e-6):
    """
    Check if trajectory samples satisfy corridor constraints (per-segment).

    sample_xz: (M,2) array of [x,z]
    segments: list from build_corridor_constraints(...)
    sample_seg: (M,) int array, segment id for each sample (qp["sample_seg"])
                If None, falls back to old behavior (NOT recommended).
    """
    sample_xz = np.asarray(sample_xz, float)
    if sample_xz.shape[0] == 0:
        return 0.0, 0, 0, 0.0

    if sample_seg is not None:
        sample_seg = np.asarray(sample_seg, int)
        if sample_seg.shape[0] != sample_xz.shape[0]:
            raise ValueError("sample_seg length must match sample_xz length")

    max_v = 0.0
    bad_count = 0
    total_count = 0

    for seg_id, seg in enumerate(segments):
        if not seg.get("use_corridor", False):
            continue
        A = seg.get("A_xz", None)
        b = seg.get("b_xz", None)
        if A is None or b is None:
            continue

        A = np.asarray(A, float)
        b = np.asarray(b, float)
        if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            continue

        # --- key fix: only check samples that belong to THIS segment ---
        if sample_seg is not None:
            mask = (sample_seg == seg_id)
            pts = sample_xz[mask]
        else:
            # fallback (old behavior): check all points against every seg
            pts = sample_xz

        if pts.shape[0] == 0:
            continue

        v = (A @ pts.T).T - b[None, :]               # (mPts, nHalf)
        v_max_per_point = np.max(v, axis=1)          # (mPts,)

        total_count += int(v_max_per_point.shape[0])
        bad_count += int(np.sum(v_max_per_point > tol))
        max_v = max(max_v, float(np.max(v_max_per_point)))

    rate = (bad_count / total_count) if total_count > 0 else 0.0
    return max_v, bad_count, total_count, rate



def union_forbidden_xz(forbidden_prisms):
    """Union forbidden polygons in XZ and fix invalid geometry."""
    if forbidden_prisms is None:
        return None
    polys = []
    for f in forbidden_prisms:
        poly = f.get("poly_xz", None)
        if poly is None:
            continue
        try:
            if poly.is_empty:
                continue
            poly = poly.buffer(0.0)  # fix invalid/self-intersection
            if poly.is_empty:
                continue
            polys.append(poly)
        except Exception:
            continue

    if not polys:
        return None

    try:
        u = unary_union(polys).buffer(0.0)
        if u.is_empty:
            return None
        return u
    except Exception:
        # fallback: iterative union
        u = polys[0]
        for p in polys[1:]:
            try:
                u = u.union(p)
            except Exception:
                pass
        try:
            u = u.buffer(0.0)
            return None if u.is_empty else u
        except Exception:
            return None


def _check_endpoints_in_corridor(segments, P_wp, tol=1e-6):
    bad = []
    for k, seg in enumerate(segments):
        A0 = seg.get("A_xz", None)
        b0 = seg.get("b_xz", None)
        if A0 is None or b0 is None:
            continue

        A = np.asarray(A0, float)
        b = np.asarray(b0, float)

        # 必须是 (m,2) 和 (m,)
        if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[0] == 0:
            continue
        if (not np.isfinite(A).all()) or (not np.isfinite(b).all()):
            continue

        i0, i1 = int(seg["i0"]), int(seg["i1"])
        p0 = np.asarray(P_wp[i0], float)[[0, 2]]
        p1 = np.asarray(P_wp[i1], float)[[0, 2]]

        v0 = A @ p0 - b
        v1 = A @ p1 - b
        m0 = float(np.max(v0)) if v0.size else 0.0
        m1 = float(np.max(v1)) if v1.size else 0.0
        if m0 > tol or m1 > tol:
            bad.append((k, i0, i1, m0, m1))
    return bad


def _check_halfspaces_numeric(segments):
    bad = []
    for k, seg in enumerate(segments):
        A0 = seg.get("A_xz", None)
        b0 = seg.get("b_xz", None)
        if A0 is None or b0 is None:
            continue

        A = np.asarray(A0, float)
        b = np.asarray(b0, float)
        if (A.ndim != 2) or (b.ndim != 1):
            bad.append((k, True, False, True, False, A.shape, b.shape))
            continue
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            bad.append((k, np.isnan(A).any(), np.isinf(A).any(),
                        np.isnan(b).any(), np.isinf(b).any(),
                        A.shape, b.shape))
    return bad


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
    corridor_margin: float = 0.02,
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

    # --- sanitize corridor constraints (never hard-crash on bad geometry) ---
    bad_num = _check_halfspaces_numeric(segments)
    if bad_num:
        print(f"[corridor][sanitize] non-finite halfspace -> disable segments={bad_num[:10]}")
        for (k, *_rest) in bad_num:
            segments[k]["A_xz"] = None
            segments[k]["b_xz"] = None
            segments[k]["use_corridor"] = False

    bad_ep = _check_endpoints_in_corridor(segments, P_wp, tol=corridor_margin)
    if bad_ep:
        print(f"[corridor][sanitize] endpoints outside corridor -> disable segments={bad_ep[:10]}")
        for (k, *_rest) in bad_ep:
            segments[k]["A_xz"] = None
            segments[k]["b_xz"] = None
            segments[k]["use_corridor"] = False


    bad_ep = _check_endpoints_in_corridor(segments, P_wp, tol=corridor_margin)
    if bad_ep:
        print(f"[corridor][sanitize] endpoints outside corridor (k,i0,i1,maxV0,maxV1) first10={bad_ep[:10]}")
        # 先别直接 raise，方便你试放宽
        # raise RuntimeError("Some hard endpoints are outside corridor => infeasible.")


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
        return slice(s, s + n)

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
                uinq_list.append(float(b_xz[r_h] + corridor_margin))

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
        print(f"[qp][setup] dim={dim} segments={K} eq={n_eq} inq={n_inq}")

    # --- Solve with OSQP ---
    prob = osqp.OSQP()
    prob.setup(P=Pmat, q=q, A=Aall, l=lall, u=uall,
               verbose=debug, polish=True, eps_abs=1e-3, eps_rel=1e-3,
               max_iter=60000)
    res = prob.solve()

    if res.info.status_val not in (1, 2, 7):  # 1=solved, 2=solved inaccurate, 7=max iter
        raise RuntimeError(f"OSQP failed: {res.info.status} (val={res.info.status_val})")
    if res.info.status_val == 7 and verbose:
        print(f"[qp3d][warn] OSQP hit max_iter but returning last iterate (status={res.info.status})")

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


def ensure_polygon(geom):
    """
    Convert shapely geometry to a single Polygon if possible.
    - If Polygon: return it
    - If MultiPolygon: return largest area polygon
    - If GeometryCollection: pick largest-area Polygon (or polygon from convex_hull)
    - If empty/None: return None
    """
    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        pass

    gtype = getattr(geom, "geom_type", None)

    if gtype == "Polygon":
        return geom

    if gtype == "MultiPolygon":
        polys = list(geom.geoms)
        polys = [p for p in polys if (p is not None and (not p.is_empty))]
        if len(polys) == 0:
            return None
        return max(polys, key=lambda p: p.area)

    if gtype == "GeometryCollection":
        polys = []
        for gg in list(geom.geoms):
            if gg is None:
                continue
            if getattr(gg, "is_empty", False):
                continue
            if getattr(gg, "geom_type", None) == "Polygon":
                polys.append(gg)
            elif getattr(gg, "geom_type", None) == "MultiPolygon":
                polys.extend(list(gg.geoms))
        polys = [p for p in polys if (p is not None and (not p.is_empty))]
        if len(polys) > 0:
            return max(polys, key=lambda p: p.area)
        # fallback: convex hull might give a Polygon/LineString/Point
        hull = geom.convex_hull
        return ensure_polygon(hull)

    # fallback: try convex hull
    hull = geom.convex_hull
    if getattr(hull, "geom_type", None) == "Polygon":
        return hull
    return None


# ---------- 1) 分段：按 hard 点切段 ----------

def split_segments_by_hard(hard: np.ndarray):
    """Return list of (i0, i1) indices for consecutive hard points."""
    hard_idx = np.where(hard)[0].astype(int)
    return [(int(a), int(b)) for a, b in zip(hard_idx[:-1], hard_idx[1:])]

# ---------- 2) forbidden 的 XZ union ----------

# ---------- 3) corridor polygon：对折线 buffer + 自动缩半径 ----------

def corridor_polygon_xz(P_seg, forbidden_union_xz=None,
                        radius=0.25, radius_min=0.06, shrink=0.85,
                        join_style=2, cap_style=2):
    """
    P_seg: (K,3) points in this segment
    Return: (poly_xz, radius_used)
    """
    xz = [(float(p[0]), float(p[2])) for p in P_seg]
    if len(xz) < 2:
        return None, 0.0
    line = LineString(xz)

    r = float(radius)
    for _ in range(30):
        poly = line.buffer(r, join_style=join_style, cap_style=cap_style).buffer(0.0)
        if forbidden_union_xz is None or (not poly.intersects(forbidden_union_xz)):
            return poly, r
        r *= float(shrink)
        if r < radius_min:
            break

    poly = line.buffer(max(r, radius_min), join_style=join_style, cap_style=cap_style).buffer(0.0)
    return poly, max(r, radius_min)

# ---------- 4) polygon -> halfspace A x <= b ----------

def polygon_to_halfspace(poly, eps_edge=1e-9):
    """
    Convert a (convex) polygon to halfspace Ax <= b.
    Robust version:
      - ensure Polygon (handle MultiPolygon/GeometryCollection)
      - convex hull
      - reject degenerate (area too small)
      - handle CW/CCW orientation
      - normalize normals
      - skip near-zero edges
    Returns (A,b) or (None,None).
    """
    poly = ensure_polygon(poly)
    if poly is None:
        return None, None

    try:
        poly = poly.buffer(0.0)  # fix invalid
        if poly.is_empty:
            return None, None
    except Exception:
        return None, None

    # convexify
    poly = ensure_polygon(poly.convex_hull)
    if poly is None:
        return None, None

    try:
        if poly.area < 1e-8:
            return None, None
    except Exception:
        return None, None

    coords = np.asarray(list(poly.exterior.coords)[:-1], dtype=float)  # (m,2)
    m = coords.shape[0]
    if m < 3:
        return None, None

    # signed area to determine orientation (CCW positive)
    x = coords[:, 0]
    y = coords[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    ccw = area2 > 0.0

    A = []
    b = []

    for i in range(m):
        p0 = coords[i]
        p1 = coords[(i + 1) % m]
        e = p1 - p0
        elen = float(np.linalg.norm(e))
        if elen < eps_edge:
            continue

        # For CCW polygon, outward normal is right-normal: [e_y, -e_x]
        # For CW polygon, outward normal is left-normal: [-e_y, e_x]
        if ccw:
            n = np.array([e[1], -e[0]], dtype=float)
        else:
            n = np.array([-e[1], e[0]], dtype=float)

        n_norm = float(np.linalg.norm(n))
        if n_norm < eps_edge:
            continue
        n = n / n_norm

        A.append(n)
        b.append(float(n.dot(p0)))

    if len(A) < 3:
        return None, None

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if (not np.isfinite(A).all()) or (not np.isfinite(b).all()):
        return None, None

    return A, b


# ---------- 5) 主入口：给定 keyframes，生成每段 corridor + halfspace ----------

def visualize_corridor_and_trajectory(segments, sample_xz, P_wp, forbidden_prisms=None, 
                                      out_path="corridor_traj_viz.png"):
    """
    Visualize XZ plane: corridor halfspaces, sampled trajectory, keyframes, forbidden areas.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import Polygon as ShapelyPolygon
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 1) Draw corridor halfspaces (as shaded regions)
    for seg_idx, seg in enumerate(segments):
        if not seg.get("use_corridor", False):
            continue
        A = seg.get("A_xz", None)
        b = seg.get("b_xz", None)
        if A is None or b is None:
            continue
        
        poly = seg.get("poly_xz", None)
        if poly is not None and not poly.is_empty:
            x, z = poly.exterior.xy
            ax.fill(x, z, alpha=0.15, color='green', edgecolor='green', linewidth=1)
    
    # 2) Draw trajectory samples
    if sample_xz.shape[0] > 0:
        ax.plot(sample_xz[:, 0], sample_xz[:, 1], 'b-', linewidth=2, label='QP Trajectory', alpha=0.8)
        ax.scatter(sample_xz[:, 0], sample_xz[:, 1], c='blue', s=8, alpha=0.5)
    
    # 3) Draw keyframes
    P_xz = P_wp[:, [0, 2]]
    ax.scatter(P_xz[:, 0], P_xz[:, 1], c='orange', s=50, marker='o', edgecolors='black', linewidth=1, label='Keyframes', zorder=5)
    
    # Connect keyframes
    for i in range(len(P_xz) - 1):
        ax.plot([P_xz[i, 0], P_xz[i+1, 0]], [P_xz[i, 1], P_xz[i+1, 1]], 'orange', alpha=0.3, linewidth=1)
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Z", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Corridor Constraints & QP Trajectory (XZ Plane)", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[viz][corridor_traj] saved path={out_path}")
    plt.close(fig)


def visualize_box_corridor_and_trajectory(segments, sample_xyz, P_wp,
                                          forbidden_prisms=None,
                                          out_path_xz="corridor_traj_xz.png",
                                          out_path_xy="corridor_traj_xy.png",
                                          out_path_yz="corridor_traj_yz.png",
                                          out_path_svg="corridor_traj_all.svg",
                                          show: bool = False):
    """重新绘制：XZ/XY/YZ 三张独立图片，自动按数据范围缩放，突出安全走廊，可选弹出窗口。"""
    import matplotlib.pyplot as plt

    sample_xyz = np.asarray(sample_xyz, float)
    P_wp = np.asarray(P_wp, float)

    # 收集每个平面的范围
    ext = {k: [[np.inf, -np.inf], [np.inf, -np.inf]] for k in ["xz", "xy", "yz"]}

    def _upd(plane, xs, ys):
        if len(xs) == 0:
            return
        e = ext[plane]
        e[0][0] = min(e[0][0], float(np.min(xs)))
        e[0][1] = max(e[0][1], float(np.max(xs)))
        e[1][0] = min(e[1][0], float(np.min(ys)))
        e[1][1] = max(e[1][1], float(np.max(ys)))

    # 走廊范围（AABB + OBB）
    for seg in segments:
        if not seg.get("enabled", True):
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = seg["bounds"]
        _upd("xz", [xmin, xmax], [zmin, zmax])
        _upd("xy", [xmin, xmax], [ymin, ymax])
        _upd("yz", [ymin, ymax], [zmin, zmax])
        if "center" in seg and "axis_d" in seg and "axis_u" in seg:
            c = np.asarray(seg["center"], float)
            d = np.asarray(seg["axis_d"], float)
            u = np.asarray(seg["axis_u"], float)
            ez = np.array([0.0, 0.0, 1.0], float)
            hl, rxy, rz = seg.get("hl", 0.0), seg.get("r_xy", 0.0), seg.get("r_z", 0.0)
            pts_xy, pts_yz = [], []
            for s1 in [+1, -1]:
                for s2 in [+1, -1]:
                    p_xy = c + s1 * hl * d + s2 * rxy * u
                    p_yz = c + s1 * hl * d + s2 * rz * ez
                    pts_xy.append([p_xy[0], p_xy[1]])
                    pts_yz.append([p_yz[1], p_yz[2]])
            pts_xy = np.asarray(pts_xy)
            pts_yz = np.asarray(pts_yz)
            if pts_xy.size:
                _upd("xy", pts_xy[:, 0], pts_xy[:, 1])
            if pts_yz.size:
                _upd("yz", pts_yz[:, 0], pts_yz[:, 1])

    # 障碍范围
    if forbidden_prisms:
        for f in forbidden_prisms:
            if "poly_xz" in f:
                poly = f["poly_xz"]
                geoms = getattr(poly, "geoms", [poly])
                for g in geoms:
                    if g.is_empty:
                        continue
                    bxmin, bzmin, bxmax, bzmax = g.bounds
                    y0, y1 = float(f.get("y_min", 0.0)), float(f.get("y_max", 0.0))
                    _upd("xz", [bxmin, bxmax], [bzmin, bzmax])
                    _upd("xy", [bxmin, bxmax], [y0, y1])
                    _upd("yz", [y0, y1], [bzmin, bzmax])

    # 轨迹与关键点范围
    if sample_xyz.shape[0]:
        _upd("xz", sample_xyz[:, 0], sample_xyz[:, 2])
        _upd("xy", sample_xyz[:, 0], sample_xyz[:, 1])
        _upd("yz", sample_xyz[:, 1], sample_xyz[:, 2])
    _upd("xz", P_wp[:, 0], P_wp[:, 2])
    _upd("xy", P_wp[:, 0], P_wp[:, 1])
    _upd("yz", P_wp[:, 1], P_wp[:, 2])

    # 填充边距
    for k in ext:
        for i in [0, 1]:
            lo, hi = ext[k][i]
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == np.inf or hi == -np.inf:
                ext[k][i] = [0.0, 1.0]
            else:
                span = hi - lo
                pad = 0.2 * span if span > 1e-6 else 0.5
                ext[k][i] = [lo - pad, hi + pad]

    def _convex_hull_2d(pts: np.ndarray):
        """单调链凸包，输入 (N,2) 返回按顺序的顶点。若不足3点则直接返回。"""
        pts = np.unique(pts, axis=0)
        if pts.shape[0] <= 2:
            return pts
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x,y
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        hull = np.array(lower[:-1] + upper[:-1], float)
        return hull

    def _poly_area(p):
        if p is None or p.shape[0] < 3:
            return 0.0
        x = p[:, 0]; y = p[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _proj_hull_from_obb(seg, plane: str):
        """用 OBB 的角点投影后取凸包；若退化成线，再用平面内的两轴重建矩形。"""
        if not ("center" in seg and "axis_d" in seg and "axis_u" in seg):
            return None
        c = np.asarray(seg["center"], float)
        d = np.asarray(seg["axis_d"], float)
        u = np.asarray(seg["axis_u"], float)
        ez = np.array([0.0, 0.0, 1.0], float)
        hl, rxy, rz = seg.get("hl", 0.0), seg.get("r_xy", 0.0), seg.get("r_z", 0.0)

        def proj(corners):
            if plane == "xz":
                return corners[:, [0, 2]]
            elif plane == "xy":
                return corners[:, [0, 1]]
            else:
                return corners[:, [1, 2]]

        # full 8-corner hull
        corners = []
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                for s3 in (+1, -1):
                    corners.append(c + s1 * hl * d + s2 * rxy * u + s3 * rz * ez)
        corners = np.asarray(corners)
        hull = _convex_hull_2d(proj(corners))
        if _poly_area(hull) > 1e-10:
            return hull

        # fallback: use two in-plane axes (drop the axis that is orthogonal to this plane)
        if plane == "xy":
            axes = [(d, hl), (u, rxy)]
        elif plane == "yz":
            axes = [(d, hl), (ez, rz)]
        else:  # xz
            axes = [(d, hl), (ez, rz)]
        pts = []
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                p = c + s1 * axes[0][1] * axes[0][0] + s2 * axes[1][1] * axes[1][0]
                pts.append(p)
        pts = np.asarray(pts)
        hull = _convex_hull_2d(proj(pts))
        return hull if hull is not None and hull.shape[0] >= 2 else None

    def _draw_plane(ax, plane: str):
        # 走廊投影
        for seg in segments:
            if not seg.get("enabled", True):
                continue
            xmin, xmax, ymin, ymax, zmin, zmax = seg["bounds"]

            if plane == "xz":
                hull = _proj_hull_from_obb(seg, plane)
                if hull is not None and hull.shape[0] >= 3:
                    xs, zs = np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1])
                    ax.fill(xs, zs, color="green", alpha=0.12, linewidth=0)
                    ax.plot(xs, zs, color="green", linewidth=1.2, alpha=0.8, label=None)
                else:
                    poly = seg.get("poly_xz")
                    if poly is not None and (not poly.is_empty):
                        x, z = poly.exterior.xy
                        ax.fill(x, z, color="green", alpha=0.12, linewidth=0)
                        ax.plot(x, z, color="green", linewidth=1.2, alpha=0.8, label=None)
                    else:
                        x = [xmin, xmax, xmax, xmin, xmin]
                        z = [zmin, zmin, zmax, zmax, zmin]
                        ax.fill(x, z, color="green", alpha=0.12, linewidth=0)
                        ax.plot(x, z, color="green", linewidth=1.2, alpha=0.8, label=None)
            else:  # xy or yz
                hull = _proj_hull_from_obb(seg, plane)
                if hull is not None and hull.shape[0] >= 3:
                    xs, ys = np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1])
                    ax.fill(xs, ys, color="green", alpha=0.12, linewidth=0)
                    ax.plot(xs, ys, color="green", linewidth=1.2, alpha=0.8, label=None)
                else:
                    if plane == "xy":
                        x = [xmin, xmax, xmax, xmin, xmin]
                        y = [ymin, ymin, ymax, ymax, ymin]
                        ax.fill(x, y, color="green", alpha=0.12, linewidth=0)
                        ax.plot(x, y, color="green", linewidth=1.2, alpha=0.8, label=None)
                    else:
                        y = [ymin, ymax, ymax, ymin, ymin]
                        z = [zmin, zmin, zmax, zmax, zmin]
                        ax.fill(y, z, color="green", alpha=0.12, linewidth=0)
                        ax.plot(y, z, color="green", linewidth=1.2, alpha=0.8, label=None)

        # 障碍投影
        if forbidden_prisms:
            for f in forbidden_prisms:
                if "poly_xz" in f:
                    poly = f["poly_xz"]
                    geoms = getattr(poly, "geoms", [poly])
                    for g in geoms:
                        if g.is_empty:
                            continue
                        x, z = g.exterior.xy
                        if plane == "xz":
                            ax.fill(x, z, color="red", alpha=0.15, linewidth=0)
                            ax.plot(x, z, color="red", alpha=0.6, linewidth=1.0)
                        elif plane == "xy":
                            xmin, zmin, xmax, zmax = g.bounds
                            y0, y1 = float(f.get("y_min", 0.0)), float(f.get("y_max", 0.0))
                            ax.fill([xmin, xmax, xmax, xmin, xmin], [y0, y0, y1, y1, y0], color="red", alpha=0.10, linewidth=0)
                            ax.plot([xmin, xmax, xmax, xmin, xmin], [y0, y0, y1, y1, y0], color="red", alpha=0.6, linewidth=1.0)
                        elif plane == "yz":
                            y0, y1 = float(f.get("y_min", 0.0)), float(f.get("y_max", 0.0))
                            z0, z1 = g.bounds[1], g.bounds[3]
                            ax.fill([y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color="red", alpha=0.10, linewidth=0)
                            ax.plot([y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color="red", alpha=0.6, linewidth=1.0)

        # 轨迹
        if sample_xyz.shape[0] > 0:
            if plane == "xz":
                ax.plot(sample_xyz[:, 0], sample_xyz[:, 2], "b-", linewidth=2, alpha=0.9, label="trajectory")
            elif plane == "xy":
                ax.plot(sample_xyz[:, 0], sample_xyz[:, 1], "b-", linewidth=2, alpha=0.9, label="trajectory")
            elif plane == "yz":
                ax.plot(sample_xyz[:, 1], sample_xyz[:, 2], "b-", linewidth=2, alpha=0.9, label="trajectory")

        # 关键点
        if plane == "xz":
            ax.scatter(P_wp[:, 0], P_wp[:, 2], c="orange", s=45, edgecolors="black", linewidth=0.8, label="keyframes", zorder=5)
            for i in range(P_wp.shape[0] - 1):
                ax.plot([P_wp[i, 0], P_wp[i + 1, 0]], [P_wp[i, 2], P_wp[i + 1, 2]], color="orange", alpha=0.4, linewidth=1)
            ax.set_xlabel("X"); ax.set_ylabel("Z")
        elif plane == "xy":
            ax.scatter(P_wp[:, 0], P_wp[:, 1], c="orange", s=45, edgecolors="black", linewidth=0.8, label="keyframes", zorder=5)
            for i in range(P_wp.shape[0] - 1):
                ax.plot([P_wp[i, 0], P_wp[i + 1, 0]], [P_wp[i, 1], P_wp[i + 1, 1]], color="orange", alpha=0.4, linewidth=1)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        elif plane == "yz":
            ax.scatter(P_wp[:, 1], P_wp[:, 2], c="orange", s=45, edgecolors="black", linewidth=0.8, label="keyframes", zorder=5)
            for i in range(P_wp.shape[0] - 1):
                ax.plot([P_wp[i, 1], P_wp[i + 1, 1]], [P_wp[i, 2], P_wp[i + 1, 2]], color="orange", alpha=0.4, linewidth=1)
            ax.set_xlabel("Y"); ax.set_ylabel("Z")

        ax.grid(True, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(ext[plane][0]); ax.set_ylim(ext[plane][1])
        ax.legend(loc="best")
        ax.set_title({"xz": "Corridor (XZ)", "xy": "Corridor (XY)", "yz": "Corridor (YZ)"}[plane])

    # 单独输出三张图
    fig_xz, ax_xz = plt.subplots(1, 1, figsize=(7, 6))
    _draw_plane(ax_xz, "xz")
    fig_xz.tight_layout(); fig_xz.savefig(out_path_xz, dpi=220, bbox_inches="tight")

    fig_xy, ax_xy = plt.subplots(1, 1, figsize=(7, 6))
    _draw_plane(ax_xy, "xy")
    fig_xy.tight_layout(); fig_xy.savefig(out_path_xy, dpi=220, bbox_inches="tight")

    if out_path_yz is not None:
        fig_yz, ax_yz = plt.subplots(1, 1, figsize=(7, 6))
        _draw_plane(ax_yz, "yz")
        fig_yz.tight_layout(); fig_yz.savefig(out_path_yz, dpi=220, bbox_inches="tight")
        print(f"[viz][box_corridor] saved XZ={out_path_xz} XY={out_path_xy} YZ={out_path_yz}")
    else:
        print(f"[viz][box_corridor] saved XZ={out_path_xz} XY={out_path_xy}")

    # 可缩放的总览 SVG（1x3 子图）
    if out_path_svg:
        fig_all, axes = plt.subplots(1, 3, figsize=(18, 6))
        _draw_plane(axes[0], "xz")
        _draw_plane(axes[1], "xy")
        _draw_plane(axes[2], "yz")
        fig_all.tight_layout()
        fig_all.savefig(out_path_svg, dpi=300, bbox_inches="tight", format="svg")
        print(f"[viz][box_corridor] saved SVG overview={out_path_svg}")

    if show:
        plt.show()

    # 保存后再关闭，避免 show=True 时提前关闭
    plt.close(fig_xz)
    plt.close(fig_xy)
    if out_path_yz is not None:
        plt.close(fig_yz)
    if out_path_svg:
        plt.close(fig_all)


def build_corridor_constraints(P_wp, tags, hard, forbidden_prisms,
                               corridor_radius=0.25, radius_min=0.06,
                               debug=False):
    """
    Return list of segments with corridor constraints.
    If corridor construction fails for a segment, it will gracefully disable
    corridor constraints on that segment (A_xz=None, b_xz=None) instead of
    producing NaN/empty halfspaces.
    """
    forb_u = union_forbidden_xz(forbidden_prisms)
    seg_pairs = split_segments_by_hard(hard)

    segments = []
    for sid, (i0, i1) in enumerate(seg_pairs):
        idxs = np.arange(i0, i1 + 1, dtype=int)
        P_seg = P_wp[idxs]

        poly, r_used = corridor_polygon_xz(
            P_seg, forbidden_union_xz=forb_u,
            radius=corridor_radius, radius_min=radius_min
        )

        poly_use = ensure_polygon(poly)
        if poly_use is not None:
            try:
                poly_use = ensure_polygon(poly_use.buffer(0.0))
                if poly_use is not None:
                    poly_use = ensure_polygon(poly_use.convex_hull)
            except Exception:
                poly_use = None

        A, b = polygon_to_halfspace(poly_use)

        # If halfspace invalid -> disable corridor constraints for this segment
        use_corridor = True
        if A is None or b is None:
            use_corridor = False
        else:
            A = np.asarray(A, float)
            b = np.asarray(b, float)
            if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[0] < 3:
                use_corridor = False
            if use_corridor and ((not np.isfinite(A).all()) or (not np.isfinite(b).all())):
                use_corridor = False

        if not use_corridor:
            if debug:
                print(f"[corridor][build] seg={sid} corridor invalid -> disable this segment")
            A, b = None, None
            poly_use = None

        y_min = float(np.min(P_seg[:, 1]))
        y_max = float(np.max(P_seg[:, 1]))

        segments.append({
            "seg_id": sid,
            "i0": int(i0), "i1": int(i1),
            "idxs": idxs,
            "P_seg": P_seg,
            "poly_xz": poly_use,
            "radius_used": float(r_used),
            "A_xz": A, "b_xz": b,
            "use_corridor": bool(use_corridor),
            "y_min": y_min, "y_max": y_max,
        })
    return segments

import numpy as np
import scipy.sparse as sp
import osqp

# -----------------------------
# 3D BOX CORRIDOR (AABB) UTILS
# -----------------------------

def _prism_aabb(prism: dict):
    """Extract AABB from your forbidden prism dict.
    Expected keys (adjust if your project uses different names):
      x_min, x_max, y_min, y_max, z_min, z_max
    """
    return (float(prism["x_min"]), float(prism["x_max"]),
            float(prism["y_min"]), float(prism["y_max"]),
            float(prism["z_min"]), float(prism["z_max"]))


def _obb_halfspaces(center, axis_d, axis_u, hl, r_xy, r_z):
    """Return A(6x3), b(6,) for oriented tube box using axes (d,u,ez).
    Enforces |d·(p-c)|<=hl, |u·(p-c)|<=r_xy, |ez·(p-c)|<=r_z.
    """
    c = np.asarray(center, float).reshape(3)
    d = np.asarray(axis_d, float).reshape(3)
    u = np.asarray(axis_u, float).reshape(3)
    ez = np.array([0.0, 0.0, 1.0], float)

    axes = [(d, hl), (u, r_xy), (ez, r_z)]
    A_list, b_list = [], []
    for a, bound in axes:
        a = np.asarray(a, float)
        A_list.append(+a)
        b_list.append(float(a @ c + bound))
        A_list.append(-a)
        b_list.append(float((-a) @ c + bound))
    return np.vstack(A_list), np.asarray(b_list)


def _aabb_intersect(a, b, eps=1e-9):
    """a,b = (xmin,xmax,ymin,ymax,zmin,zmax)"""
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    if ax1 <= bx0 + eps or bx1 <= ax0 + eps: return False
    if ay1 <= by0 + eps or by1 <= ay0 + eps: return False
    if az1 <= bz0 + eps or bz1 <= az0 + eps: return False
    return True


import numpy as np
from shapely.geometry import LineString

def build_box_corridor_constraints(
    P_wp: np.ndarray,
    hard: np.ndarray,
    tags: np.ndarray,
    door_ids: np.ndarray,
    forbidden_prisms: list,
    margin_xz: float = 0.25,
    margin_y: float = 0.25,
    shrink: float = 0.7,
    max_shrink_iter: int = 12,
    min_radius: float = 0.02,
    pad_long: float = 0.05,
):
    """
    Build per-segment TUBE corridor:
    - In XZ: tube polygon = buffer(polyline, r), then convex_hull -> halfspace A_xz, b_xz
    - In Y : [ymin - my, ymax + my]
    - Additionally build a 3D oriented box (axis d along segment, side axis from global z) with halfspaces A_obb, b_obb
    Also checks tube-vs-obstacle intersection in XZ (with Y overlap).
    Returns segments list, each:
      - enabled: bool
      - y_bounds: (y0,y1)
      - A_xz, b_xz: halfspace for [x,z]
      - poly_xz: shapely polygon (for debug/plot)
      - idx_range: (i0,i1)
      - r_used, my_used
    """
    from shapely.geometry import Polygon

    P_wp = np.asarray(P_wp, float)
    hard = np.asarray(hard, bool)
    hard_idx = np.where(hard)[0]
    assert hard_idx.size >= 2

    # precompute obstacle xz polygons + y slab
    forb_polys = []
    for prism in forbidden_prisms:
        # prism assumed to be (xmin,xmax,ymin,ymax,zmin,zmax) or dict-like -> adapt if needed
        if isinstance(prism, dict):
            xmin, xmax = prism["xmin"], prism["xmax"]
            ymin, ymax = prism["ymin"], prism["ymax"]
            zmin, zmax = prism["zmin"], prism["zmax"]
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = prism
        poly = Polygon([(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)])
        forb_polys.append((poly, float(ymin), float(ymax)))

    segments = []
    for s in range(hard_idx.size - 1):
        i0 = int(hard_idx[s])
        i1 = int(hard_idx[s + 1])
        seg_pts = P_wp[i0:i1 + 1]
        if seg_pts.shape[0] < 2:
            # degenerate
            p = seg_pts[0]
            poly = Polygon([(p[0], p[2]), (p[0], p[2]), (p[0], p[2])]).buffer(1e-3).convex_hull
            A, b = convex_polygon_to_halfspace(poly)
            xmin = xmax = float(p[0])
            zmin = zmax = float(p[2])
            ymin = ymax = float(p[1])
            d = np.array([1.0, 0.0, 0.0])
            u = np.array([0.0, 1.0, 0.0])
            hl = 1e-3
            A3, b3 = _obb_halfspaces(p, d, u, hl, r_xy=margin_xz, r_z=margin_y)
            segments.append({
                "enabled": True,
                "y_bounds": (ymin, ymax),
                "A_xz": A,
                "b_xz": b,
                "A_obb": A3,
                "b_obb": b3,
                "axis_d": d,
                "axis_u": u,
                "center": p,
                "hl": hl,
                "r_xy": float(margin_xz),
                "r_z": float(margin_y),
                "poly_xz": poly,
                "bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
                "idx_range": (i0, i1),
                "r_used": 0.0,
                "my_used": 0.0,
            })
            continue

        # y slab from points
        ymin_base = float(np.min(seg_pts[:, 1]))
        ymax_base = float(np.max(seg_pts[:, 1]))

        r = float(margin_xz)
        my = float(margin_y)

        xz_line = LineString(seg_pts[:, [0, 2]])

        enabled = True
        poly_ok = None
        A_ok, b_ok = None, None
        y0_ok, y1_ok = None, None
        r_final = None
        my_final = None

        for _ in range(max_shrink_iter + 1):
            if r < min_radius:
                break

            tube = xz_line.buffer(r, cap_style=2, join_style=2).buffer(0.0)
            hull = tube.convex_hull  # make it convex for halfspace

            y0 = ymin_base - my
            y1 = ymax_base + my

            # obstacle intersection check: only if y slabs overlap
            hit = False
            for poly_obs, oy0, oy1 in forb_polys:
                if (y1 < oy0) or (y0 > oy1):
                    continue
                if hull.intersects(poly_obs):
                    hit = True
                    break

            A, b = convex_polygon_to_halfspace(hull)
            if (A is None) or (b is None):
                hit = True  # treat as bad geometry

            if not hit:
                poly_ok = hull
                A_ok, b_ok = A, b
                y0_ok, y1_ok = y0, y1
                r_final = r
                my_final = my
                break

            # shrink
            r *= float(shrink)
            my *= float(shrink)

        if poly_ok is None:
            # cannot build a safe tube
            enabled = False
            # still output something for debug, but mark disabled
            tube = xz_line.buffer(max(r, min_radius), cap_style=2, join_style=2).buffer(0.0)
            hull = tube.convex_hull
            A, b = convex_polygon_to_halfspace(hull)
            poly_ok = hull
            A_ok, b_ok = A, b
            y0_ok, y1_ok = ymin_base, ymax_base
            r_final = r
            my_final = my

        xmin, zmin, xmax, zmax = poly_ok.bounds
        # build 3D oriented box using segment end direction
        seg_dir = seg_pts[-1] - seg_pts[0]
        L = np.linalg.norm(seg_dir)
        if L < 1e-9:
            seg_dir = np.array([1.0, 0.0, 0.0], float)
            L = 1e-9
        d_axis = seg_dir / L
        # side axis from global up to keep vertical intuitive
        up = np.array([0.0, 0.0, 1.0], float)
        side = np.cross(up, d_axis)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([0.0, 1.0, 0.0], float)
        side = side / (np.linalg.norm(side) + 1e-12)
        center = 0.5 * (seg_pts[0] + seg_pts[-1])
        hl = 0.5 * L + float(pad_long)
        widen = 1.3
        r_lat = float(widen * (r_final if r_final is not None else r) + 0.02)
        r_vert = float(widen * (my_final if my_final is not None else my) + 0.02)
        A3, b3 = _obb_halfspaces(center, d_axis, side, hl, r_lat, r_vert)

        segments.append({
            "enabled": bool(enabled),
            "y_bounds": (float(y0_ok), float(y1_ok)),
            "A_xz": A_ok,
            "b_xz": b_ok,
            "poly_xz": poly_ok,
            "A_obb": A3,
            "b_obb": b3,
            "axis_d": d_axis,
            "axis_u": side,
            "center": center,
            "hl": hl,
            "r_xy": r_lat,
            "r_z": r_vert,
            "bounds": (float(xmin), float(xmax), float(y0_ok), float(y1_ok), float(zmin), float(zmax)),
            "idx_range": (i0, i1),
            "r_used": float(r_lat),
            "my_used": float(r_vert),
        })

    return segments



def check_traj_in_box_corridor(sample_xyz, segments, sample_seg=None, tol=1e-6):
    """Check sampled 3D trajectory against per-segment box corridors.

    sample_xyz: (M,3) samples.
    segments: from build_box_corridor_constraints (each has bounds, enabled flag).
    sample_seg: (M,) segment id per sample; if None, all samples are checked against every enabled box.
    tol: allowed slack; violation > tol counts as outside.

    Returns (max_violation, bad_count, total_count, rate).
    """
    pts = np.asarray(sample_xyz, float)
    if pts.shape[0] == 0:
        return 0.0, 0, 0, 0.0

    if sample_seg is not None:
        sample_seg = np.asarray(sample_seg, int)
        if sample_seg.shape[0] != pts.shape[0]:
            raise ValueError("sample_seg length must match sample_xyz length")

    max_v = 0.0
    bad = 0
    total = 0

    for sid, seg in enumerate(segments):
        if not seg.get("enabled", True):
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = seg["bounds"]

        if sample_seg is not None:
            mask = (sample_seg == sid)
            cur = pts[mask]
        else:
            cur = pts

        if cur.shape[0] == 0:
            continue

        A_obb = seg.get("A_obb", None)
        b_obb = seg.get("b_obb", None)
        use_obb = A_obb is not None and b_obb is not None

        if use_obb:
            A_obb = np.asarray(A_obb, float)
            b_obb = np.asarray(b_obb, float)
            v = A_obb @ cur.T - b_obb.reshape(-1, 1)
            v = v.T  # shape (N,6)
        else:
            v = np.stack([
                cur[:, 0] - xmax,
                xmin - cur[:, 0],
                cur[:, 1] - ymax,
                ymin - cur[:, 1],
                cur[:, 2] - zmax,
                zmin - cur[:, 2],
            ], axis=1)

        v_max = np.max(v, axis=1)
        total += int(v_max.shape[0])
        bad += int(np.sum(v_max > tol))
        max_v = max(max_v, float(np.max(v_max)))

    rate = (bad / total) if total > 0 else 0.0
    return max_v, bad, total, rate


# -----------------------------
# 3D MIN-SNAP QP (OSQP)
# -----------------------------

def _poly_basis(t: float, order: int = 7, deriv: int = 0) -> np.ndarray:
    """Same style you already use: basis for d^deriv/dt^deriv of [1,t,t^2,...]."""
    n = order + 1
    v = np.zeros(n, dtype=float)
    for i in range(n):
        if i < deriv:
            continue
        # coefficient of i*(i-1)*...*(i-deriv+1) * t^(i-deriv)
        c = 1.0
        for k in range(deriv):
            c *= (i - k)
        v[i] = c * (t ** (i - deriv))
    return v


def _Q_snap(order: int, T: float, snap_deriv: int = 4) -> np.ndarray:
    """Integral_0^T (d^snap x / dt^snap)^2 dt => c^T Q c"""
    n = order + 1
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i < snap_deriv or j < snap_deriv:
                continue
            ci = 1.0
            cj = 1.0
            for k in range(snap_deriv):
                ci *= (i - k)
                cj *= (j - k)
            p = i + j - 2 * snap_deriv
            Q[i, j] = ci * cj * (T ** (p + 1)) / (p + 1)
    return Q


def solve_minsnap_xyz_with_box_corridors_osqp(
    P_wp: np.ndarray,
    hard: np.ndarray,
    segments: list,
    order: int = 7,
    T_per_seg: float | np.ndarray | list = 1.0,
    snap_deriv: int = 4,
    n_ineq_samples: int = 15,
    eps_corridor: float = 1e-6,
    verbose: bool = True,
):
    """
    Solve 3D min-snap polynomial trajectory with:
      - hard waypoint equality constraints at segment endpoints (x,y,z)
      - continuity constraints up to 3rd derivative at internal joints (x,y,z)
      - box corridor inequality constraints sampled in time per segment

    Decision vars:
      For each segment k: coeffs_x[k,0..n-1], coeffs_y[k,0..n-1], coeffs_z[k,0..n-1]
    """
    P_wp = np.asarray(P_wp, float)
    hard = np.asarray(hard, bool)
    hard_idx = np.where(hard)[0]
    K = hard_idx.size - 1
    assert K == len(segments), f"K mismatch: hard segments={K}, segments={len(segments)}"
    n = order + 1
    dim_seg = 3 * n
    dim = K * dim_seg

    # per-segment durations
    if np.isscalar(T_per_seg):
        T_list = np.full(K, float(T_per_seg), dtype=float)
    else:
        T_list = np.asarray(T_per_seg, float).reshape(-1)
        if T_list.size != K:
            raise ValueError(f"T_per_seg length mismatch: got {T_list.size}, expected {K}")

    # --- safety: ensure endpoints are inside corridor primitives ---
    # If OBB halfspaces exclude endpoints, drop OBB for that segment (fallback to AABB bounds).
    # Also expand AABB slightly to include endpoints if numerical issues occur.
    eps_bound = max(eps_corridor, 1e-4)
    drop_obb = []
    expand_bounds = []
    for k, seg in enumerate(segments):
        i0 = int(hard_idx[k])
        i1 = int(hard_idx[k + 1])
        p0 = np.asarray(P_wp[i0], float)
        p1 = np.asarray(P_wp[i1], float)
        xmin, xmax, ymin, ymax, zmin, zmax = seg["bounds"]

        # expand bounds if endpoints are marginally outside
        for p in (p0, p1):
            px, py, pz = p
            if px < xmin - eps_bound: xmin = float(px - eps_bound)
            if px > xmax + eps_bound: xmax = float(px + eps_bound)
            if py < ymin - eps_bound: ymin = float(py - eps_bound)
            if py > ymax + eps_bound: ymax = float(py + eps_bound)
            if pz < zmin - eps_bound: zmin = float(pz - eps_bound)
            if pz > zmax + eps_bound: zmax = float(pz + eps_bound)
        seg["bounds"] = (xmin, xmax, ymin, ymax, zmin, zmax)
        expand_bounds.append((k, xmin, xmax, ymin, ymax, zmin, zmax))

        A_obb = seg.get("A_obb", None)
        b_obb = seg.get("b_obb", None)
        if A_obb is None or b_obb is None:
            continue
        A_obb = np.asarray(A_obb, float)
        b_obb = np.asarray(b_obb, float)
        v0 = A_obb @ p0 - b_obb
        v1 = A_obb @ p1 - b_obb
        if np.max(v0) > eps_bound or np.max(v1) > eps_bound or (not np.isfinite(v0).all()) or (not np.isfinite(v1).all()):
            seg["A_obb"] = None
            seg["b_obb"] = None
            drop_obb.append(k)

    # helper: index mapping
    def idx_x(k): return k * dim_seg + 0
    def idx_y(k): return k * dim_seg + n
    def idx_z(k): return k * dim_seg + 2 * n

    # Objective P (block diagonal)
    P_blocks = []
    for _k in range(K):
        Qk = _Q_snap(order, float(T_list[_k]), snap_deriv=snap_deriv)
        Pk = sp.block_diag([Qk, Qk, Qk], format="csc")
        P_blocks.append(Pk)
    P = sp.block_diag(P_blocks, format="csc")

    q = np.zeros(dim, dtype=float)

    # Equality constraints Aeq x = beq
    Aeq_rows = []
    beq = []

    def add_eq(row_dict, rhs):
        Aeq_rows.append(row_dict)
        beq.append(float(rhs))

    # 1) Hard endpoint position constraints (x,y,z at t=0 and t=T)
    for k in range(K):
        i0 = int(hard_idx[k])
        i1 = int(hard_idx[k + 1])
        p0 = P_wp[i0]
        p1 = P_wp[i1]

        phi0 = _poly_basis(0.0, order=order, deriv=0)
        phiT = _poly_basis(float(T_list[k]), order=order, deriv=0)

        # x(t=0)=p0.x
        row = {idx_x(k) + j: phi0[j] for j in range(n)}
        add_eq(row, p0[0])
        # x(t=T)=p1.x
        row = {idx_x(k) + j: phiT[j] for j in range(n)}
        add_eq(row, p1[0])

        # y
        row = {idx_y(k) + j: phi0[j] for j in range(n)}
        add_eq(row, p0[1])
        row = {idx_y(k) + j: phiT[j] for j in range(n)}
        add_eq(row, p1[1])

        # z
        row = {idx_z(k) + j: phi0[j] for j in range(n)}
        add_eq(row, p0[2])
        row = {idx_z(k) + j: phiT[j] for j in range(n)}
        add_eq(row, p1[2])

    # 2) Continuity at internal joints up to 3rd derivative (pos/vel/acc/jerk)
    for k in range(K - 1):
        for d in [0, 1, 2, 3]:
            phiT = _poly_basis(float(T_list[k]), order=order, deriv=d)
            phi0 = _poly_basis(0.0, order=order, deriv=d)

            # x continuity
            row = {}
            for j in range(n):
                row[idx_x(k) + j] = row.get(idx_x(k) + j, 0.0) + phiT[j]
                row[idx_x(k + 1) + j] = row.get(idx_x(k + 1) + j, 0.0) - phi0[j]
            add_eq(row, 0.0)

            # y continuity
            row = {}
            for j in range(n):
                row[idx_y(k) + j] = row.get(idx_y(k) + j, 0.0) + phiT[j]
                row[idx_y(k + 1) + j] = row.get(idx_y(k + 1) + j, 0.0) - phi0[j]
            add_eq(row, 0.0)

            # z continuity
            row = {}
            for j in range(n):
                row[idx_z(k) + j] = row.get(idx_z(k) + j, 0.0) + phiT[j]
                row[idx_z(k + 1) + j] = row.get(idx_z(k + 1) + j, 0.0) - phi0[j]
            add_eq(row, 0.0)

    # Optional: boundary derivatives at start/end = 0 (v,a,j)
    # 你后面要更真实的动力学，这里可换成约束初末速度/加速度
    for d in [1, 2, 3]:
        phi0 = _poly_basis(0.0, order=order, deriv=d)
        # start segment k=0 at t=0
        for base in [idx_x(0), idx_y(0), idx_z(0)]:
            row = {base + j: phi0[j] for j in range(n)}
            add_eq(row, 0.0)
        # end segment k=K-1 at t=T
        phiT_end = _poly_basis(float(T_list[K - 1]), order=order, deriv=d)
        for base in [idx_x(K - 1), idx_y(K - 1), idx_z(K - 1)]:
            row = {base + j: phiT_end[j] for j in range(n)}
            add_eq(row, 0.0)

    # Build sparse Aeq
    m_eq = len(Aeq_rows)
    Aeq = sp.lil_matrix((m_eq, dim), dtype=float)
    for r, row in enumerate(Aeq_rows):
        for c, v in row.items():
            Aeq[r, c] = v
    Aeq = Aeq.tocsc()
    beq = np.asarray(beq, float)

    # Inequality constraints: oriented box corridor (preferred) or AABB sampled in time
    Aiq_rows = []
    uiq = []

    def add_ineq(row_dict, ub):
        Aiq_rows.append(row_dict)
        uiq.append(float(ub))

    for k, seg in enumerate(segments):
        if not seg.get("enabled", True):
            continue
        ts = np.linspace(0.0, float(T_list[k]), n_ineq_samples)

        A_obb = seg.get("A_obb", None)
        b_obb = seg.get("b_obb", None)

        if A_obb is not None and b_obb is not None:
            A_obb = np.asarray(A_obb, float)
            b_obb = np.asarray(b_obb, float)
        else:
            A_obb = b_obb = None

        use_obb = (A_obb is not None) and A_obb.ndim == 2 and A_obb.shape[1] == 3 and b_obb.shape[0] == A_obb.shape[0]

        xmin, xmax, ymin, ymax, zmin, zmax = seg["bounds"]

        for t in ts:
            phi = _poly_basis(float(t), order=order, deriv=0)

            if use_obb:
                # each halfspace a^T p <= b
                for r_h in range(A_obb.shape[0]):
                    ax, ay, az = map(float, A_obb[r_h])
                    row = {idx_x(k) + j: ax * phi[j] for j in range(n)}
                    for j in range(n):
                        row[idx_y(k) + j] = row.get(idx_y(k) + j, 0.0) + ay * phi[j]
                        row[idx_z(k) + j] = row.get(idx_z(k) + j, 0.0) + az * phi[j]
                    add_ineq(row, float(b_obb[r_h] + eps_corridor))
            else:
                # fallback to axis-aligned bounds
                # x <= xmax
                row = {idx_x(k) + j: phi[j] for j in range(n)}
                add_ineq(row, xmax + eps_corridor)
                # -x <= -xmin
                row = {idx_x(k) + j: -phi[j] for j in range(n)}
                add_ineq(row, -xmin + eps_corridor)

                # y <= ymax
                row = {idx_y(k) + j: phi[j] for j in range(n)}
                add_ineq(row, ymax + eps_corridor)
                # -y <= -ymin
                row = {idx_y(k) + j: -phi[j] for j in range(n)}
                add_ineq(row, -ymin + eps_corridor)

                # z <= zmax
                row = {idx_z(k) + j: phi[j] for j in range(n)}
                add_ineq(row, zmax + eps_corridor)
                # -z <= -zmin
                row = {idx_z(k) + j: -phi[j] for j in range(n)}
                add_ineq(row, -zmin + eps_corridor)

    m_iq = len(Aiq_rows)
    Aiq = sp.lil_matrix((m_iq, dim), dtype=float)
    for r, row in enumerate(Aiq_rows):
        for c, v in row.items():
            Aiq[r, c] = v
    Aiq = Aiq.tocsc()
    uiq = np.asarray(uiq, float)

    # OSQP expects l <= A x <= u
    # For equalities: l=u=beq
    # For inequalities: l=-inf, u=uiq
    A = sp.vstack([Aeq, Aiq], format="csc")
    l = np.hstack([beq, -np.inf * np.ones(m_iq)])
    u = np.hstack([beq, uiq])

    if verbose:
        if drop_obb:
            print(f"[qp3d][sanitize] dropped OBB on segments {drop_obb[:10]} due to endpoint violation")
        print(f"[qp3d][setup] dim={dim} K={K} eq={m_eq} inq={m_iq}")

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=verbose, max_iter=20000, polish=True, warm_start=True)
    res = prob.solve()

    if res.info.status not in ("solved", "solved inaccurate"):
        raise RuntimeError(f"OSQP failed: {res.info.status} (val={res.info.status_val})")

    x = res.x

    coeffs_x = np.zeros((K, n), dtype=float)
    coeffs_y = np.zeros((K, n), dtype=float)
    coeffs_z = np.zeros((K, n), dtype=float)
    for k in range(K):
        coeffs_x[k] = x[idx_x(k):idx_x(k) + n]
        coeffs_y[k] = x[idx_y(k):idx_y(k) + n]
        coeffs_z[k] = x[idx_z(k):idx_z(k) + n]
    # Sample trajectory for visualization/validation
    sample_t = []
    sample_xyz = []
    sample_seg = []
    t_global = 0.0
    dt = 0.08
    for k in range(K):
        Tk = float(T_list[k])
        m = max(2, int(np.ceil(Tk / dt)))
        ts = np.linspace(0.0, Tk, m)
        for tt in ts:
            phi = _poly_basis(float(tt), order=order, deriv=0)
            xx = float(phi @ coeffs_x[k])
            yy = float(phi @ coeffs_y[k])
            zz = float(phi @ coeffs_z[k])
            sample_t.append(t_global + float(tt))
            sample_xyz.append([xx, yy, zz])
            sample_seg.append(k)
        t_global += Tk

    return {
        "status": res.info.status,
        "obj": float(res.info.obj_val),
        "coeffs_x": coeffs_x,
        "coeffs_y": coeffs_y,
        "coeffs_z": coeffs_z,
        "T_per_seg": np.asarray(T_list, float),
        "order": int(order),
        "hard_idx": hard_idx,
        "segments": segments,
        "sample_t": np.asarray(sample_t, float),
        "sample_xyz": np.asarray(sample_xyz, float),
        "sample_seg": np.asarray(sample_seg, int),
    }


def solve_minsnap_xyz_with_tube_corridors_osqp(
    P_wp: np.ndarray,
    hard: np.ndarray,
    segments: list,
    order: int = 7,
    T_per_seg: float = 1.0,
    snap_deriv: int = 4,
    n_ineq_samples: int = 15,
    eps_corridor: float = 1e-6,
    verbose: bool = True,
):
    """
    3D min-snap with:
      - hard waypoint equality constraints at segment endpoints
      - continuity up to 3rd derivative
      - tube corridor inequalities:
          A_xz @ [x(t), z(t)] <= b_xz - eps
          y0 <= y(t) <= y1
    """
    import scipy.sparse as sp
    import osqp

    P_wp = np.asarray(P_wp, float)
    hard = np.asarray(hard, bool)

    hard_idx = np.where(hard)[0]
    K = hard_idx.size - 1
    if K <= 0:
        raise ValueError("need at least 2 hard points")

    n = order + 1
    dim = 3 * K * n  # x,y,z for each segment

    def idx_x(k): return (k * 3 + 0) * n
    def idx_y(k): return (k * 3 + 1) * n
    def idx_z(k): return (k * 3 + 2) * n

    # -------- objective P (block diag of snap Q) ----------
    Qs = _Q_snap(order, T_per_seg, snap_deriv)
    P_blocks = []
    for _ in range(3 * K):
        P_blocks.append(Qs)
    P = sp.block_diag(P_blocks, format="csc")

    # -------- constraints A x = b  and  A x <= u ----------
    Aeq_rows = []
    beq = []

    def add_eq_row(row, rhs):
        Aeq_rows.append(row)
        beq.append(float(rhs))

    # endpoint + continuity constraints
    # for each segment k:
    #   pos(0)=P[i0], pos(T)=P[i1]
    # and continuity at joints for derivatives 0..3
    for k in range(K):
        i0 = int(hard_idx[k])
        i1 = int(hard_idx[k + 1])
        p0 = P_wp[i0]
        p1 = P_wp[i1]

        # pos at t=0
        phi0 = _poly_basis(0.0, order, 0)
        row = np.zeros(dim); row[idx_x(k):idx_x(k)+n] = phi0
        add_eq_row(row, p0[0])
        row = np.zeros(dim); row[idx_y(k):idx_y(k)+n] = phi0
        add_eq_row(row, p0[1])
        row = np.zeros(dim); row[idx_z(k):idx_z(k)+n] = phi0
        add_eq_row(row, p0[2])

        # pos at t=T
        phiT = _poly_basis(T_per_seg, order, 0)
        row = np.zeros(dim); row[idx_x(k):idx_x(k)+n] = phiT
        add_eq_row(row, p1[0])
        row = np.zeros(dim); row[idx_y(k):idx_y(k)+n] = phiT
        add_eq_row(row, p1[1])
        row = np.zeros(dim); row[idx_z(k):idx_z(k)+n] = phiT
        add_eq_row(row, p1[2])

        # continuity with next segment at joint (k < K-1)
        if k < K - 1:
            for d in range(1, 4):  # vel, acc, jerk
                phi_end = _poly_basis(T_per_seg, order, d)
                phi_next0 = _poly_basis(0.0, order, d)

                row = np.zeros(dim)
                row[idx_x(k):idx_x(k)+n] = phi_end
                row[idx_x(k+1):idx_x(k+1)+n] = -phi_next0
                add_eq_row(row, 0.0)

                row = np.zeros(dim)
                row[idx_y(k):idx_y(k)+n] = phi_end
                row[idx_y(k+1):idx_y(k+1)+n] = -phi_next0
                add_eq_row(row, 0.0)

                row = np.zeros(dim)
                row[idx_z(k):idx_z(k)+n] = phi_end
                row[idx_z(k+1):idx_z(k+1)+n] = -phi_next0
                add_eq_row(row, 0.0)

    Aeq = sp.csc_matrix(np.vstack(Aeq_rows)) if Aeq_rows else sp.csc_matrix((0, dim))
    leq = np.asarray(beq, float)
    ueq = np.asarray(beq, float)

    # -------- inequality constraints: tube corridor sampling ----------
    Ainq_rows = []
    linq = []
    uinq = []

    def add_leq_row(row, upper):
        Ainq_rows.append(row)
        linq.append(-np.inf)
        uinq.append(float(upper))

    def add_geq_row(row, lower):
        Ainq_rows.append(row)
        linq.append(float(lower))
        uinq.append(np.inf)

    ts = np.linspace(0.0, T_per_seg, n_ineq_samples)

    for k in range(K):
        seg = segments[k]
        if not seg.get("enabled", True):
            continue  # disabled corridor: do not add ineq here (we'll handle by promoting hard points outside)

        A_xz = seg.get("A_xz", None)
        b_xz = seg.get("b_xz", None)
        y0, y1 = seg.get("y_bounds", (None, None))
        if A_xz is None or b_xz is None:
            continue

        A_xz = np.asarray(A_xz, float)
        b_xz = np.asarray(b_xz, float)

        for t in ts:
            phi = _poly_basis(t, order, 0)

            # y bounds
            rowy = np.zeros(dim); rowy[idx_y(k):idx_y(k)+n] = phi
            add_leq_row(rowy, y1 - eps_corridor)
            add_geq_row(rowy, y0 + eps_corridor)

            # halfspaces on xz
            # a1*x(t)+a2*z(t) <= b
            for j in range(A_xz.shape[0]):
                a1, a2 = float(A_xz[j, 0]), float(A_xz[j, 1])
                bj = float(b_xz[j]) - eps_corridor
                row = np.zeros(dim)
                row[idx_x(k):idx_x(k)+n] = a1 * phi
                row[idx_z(k):idx_z(k)+n] = a2 * phi
                add_leq_row(row, bj)

    Ainq = sp.csc_matrix(np.vstack(Ainq_rows)) if Ainq_rows else sp.csc_matrix((0, dim))
    linq = np.asarray(linq, float) if linq else np.zeros((0,), float)
    uinq = np.asarray(uinq, float) if uinq else np.zeros((0,), float)

    # stack A
    A = sp.vstack([Aeq, Ainq], format="csc")
    l = np.concatenate([leq, linq])
    u = np.concatenate([ueq, uinq])

    if verbose:
        print(f"[qp3d_tube][setup] dim={dim} K={K} eq={Aeq.shape[0]} inq={Ainq.shape[0]}")

    q = np.zeros(dim, dtype=float)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=verbose, eps_abs=1e-3, eps_rel=1e-3, max_iter=80000)
    res = prob.solve()

    if res.info.status not in ("solved", "solved inaccurate"):
        raise RuntimeError(f"OSQP failed: {res.info.status} (val={res.info.status_val})")

    x = res.x

    # unpack coeffs
    coeffs_x = np.zeros((K, n))
    coeffs_y = np.zeros((K, n))
    coeffs_z = np.zeros((K, n))
    for k in range(K):
        coeffs_x[k] = x[idx_x(k):idx_x(k)+n]
        coeffs_y[k] = x[idx_y(k):idx_y(k)+n]
        coeffs_z[k] = x[idx_z(k):idx_z(k)+n]

    return {
        "K": K,
        "order": order,
        "T": T_per_seg,
        "hard_idx": hard_idx,
        "coeffs_x": coeffs_x,
        "coeffs_y": coeffs_y,
        "coeffs_z": coeffs_z,
        "osqp_status": res.info.status,
        "osqp_obj": float(res.info.obj_val),
        "res": res,
    }


def visualize_keyframes_detail(P_wp, hard, tags, door_ids, out_path="keyframes_detail.png", plane="xz"):
    """可视化关键点，区分硬约束点(大星号)和软约束点(小圆点)，显示序号."""
    import matplotlib.pyplot as plt
    
    P_wp = np.asarray(P_wp, float)
    hard = np.asarray(hard, bool)
    tags = np.asarray(tags, dtype=object)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 选择投影
    if plane == "xz":
        axis0, axis1 = 0, 2
        xlabel, ylabel = "X", "Z"
    elif plane == "xy":
        axis0, axis1 = 0, 1
        xlabel, ylabel = "X", "Y"
    else:  # yz
        axis0, axis1 = 1, 2
        xlabel, ylabel = "Y", "Z"
    
    pts_hard = P_wp[hard]
    pts_soft = P_wp[~hard]
    
    # 绘制硬约束点（大星号，红色）
    if pts_hard.shape[0] > 0:
        ax.scatter(pts_hard[:, axis0], pts_hard[:, axis1], s=200, marker='*', 
                  c='red', edgecolors='darkred', linewidth=1.5, label='Hard (pre/mid/post)', zorder=5)
    
    # 绘制软约束点（小圆点，绿色）
    if pts_soft.shape[0] > 0:
        ax.scatter(pts_soft[:, axis0], pts_soft[:, axis1], s=60, marker='o', 
                  c='green', edgecolors='darkgreen', linewidth=0.8, alpha=0.7, label='Soft (A*)', zorder=4)
    
    # 连接所有点
    pts_proj = P_wp[:, [axis0, axis1]]
    ax.plot(pts_proj[:, 0], pts_proj[:, 1], 'gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # 标注点序号和标签
    for i, (p, tag, is_hard, did) in enumerate(zip(P_wp, tags, hard, door_ids)):
        x, y = float(p[axis0]), float(p[axis1])
        # 点的序号
        ax.text(x, y - 0.08, f"{i}", fontsize=8, ha='center', va='top', zorder=3)
        # 标签（仅硬约束点标注）
        if is_hard:
            label_str = tag
            if tag == "door_mid" or tag == "door_pre" or tag == "door_post":
                label_str = f"{tag[:7]}"  # "door_mi", "door_pr", "door_po"
            ax.text(x + 0.1, y, label_str, fontsize=7, ha='left', zorder=3, style='italic')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"Keyframes Detail ({plane.upper()}): Hard vs Soft", fontsize=13, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[viz][keyframes] saved {plane}={out_path}")
    plt.close(fig)
