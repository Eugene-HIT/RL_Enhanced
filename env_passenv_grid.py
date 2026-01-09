# env_passenv_grid.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Polygon, Point

from geom_encoder import polygon_to_grid
from tools.geom_random import sample_random_door_polygon

from shapely.affinity import rotate



class PassEnvGrid(gym.Env):
    """
    Observation:
      - grid (16x16 p_free) flatten -> 256
      - door center (cx,cz)         -> 2
      - payload dims (L_eff,H_eff)  -> 2
      - theta_prev_norm            -> 1
      total = 261
    Action:
      - ax, az, atheta in [-1,1]
    Single-step episode.
    """

    def __init__(self, grid_n=16, grid_samples=3):
        super().__init__()

        self.grid_n = int(grid_n)
        self.grid_samples = int(grid_samples)

        # payload dims (keep consistent with your original env)
        self.L_eff = 0.6
        self.H_eff = 0.2

        # action: [ax, az, atheta]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # theta limits
        self.theta_max_deg = 80.0
        self.theta_max_rad = np.deg2rad(self.theta_max_deg)

        # reward weights (starter defaults)
        self.k_margin = 7.0
        self.w_pos    = 0.25
        self.w_theta  = 0.6
        self.w_dtheta = 0.25
        self.w_low    = 0.2


        # episode
        self.max_steps = 1
        self.step_count = 0

        # door cache
        self.door_poly = None
        self.door_bbox = None  # (xmin,zmin,xmax,zmax)
        self.door_center = None  # (cx,cz)
        self.shape_type = None
        self._grid_cached = None

        # initial attitude prior (randomized each episode)
        self.theta_prev = 0.0

        # obs dim
        self.obs_dim = self.grid_n * self.grid_n + 2 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # ---- randomize theta_prev (IMPORTANT: also exposed in obs) ----
        self.theta_prev = float(self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))

        # door center (keep consistent)
        door_cx0 = 1.2
        door_cz0 = 1.0 + float(self.np_random.uniform(-0.1, 0.1))
        door_center = (door_cx0, door_cz0)

        # ----------------------------------------------------------
        # choose source: template(0-3) or random polygon
        # default: mixed, e.g. 70% template + 30% random (you can tune)
        # ----------------------------------------------------------
        force_shape = None
        force_random = False
        if isinstance(options, dict):
            force_shape = options.get("shape_type", None)     # 0~3 -> template
            force_random = bool(options.get("use_random", False))

        if force_random:
            use_random = True
        elif force_shape is not None:
            use_random = False
            self.shape_type = int(force_shape)
        else:
            # 混合比例：建议先模板为主（更“有序”），随机为辅
            # 例如：0.7 -> 70%模板，30%随机
            use_random = (float(self.np_random.random()) < 0.30)
            if not use_random:
                self.shape_type = int(self.np_random.integers(0, 4))

        # ----------------------------------------------------------
        # build polygon
        # ----------------------------------------------------------
        if use_random:
            # ============ random concave/convex polygon door ============
            seed_i = int(self.np_random.integers(0, 2**31 - 1))
            poly, _verts = sample_random_door_polygon(
                seed=seed_i,
                center=door_center,
                concave_prob=0.8,
                n_vertices_range=(10, 18),
                n_notches_range=(0, 3),
                scale_x_range=(0.9, 1.8),
                scale_z_range=(0.9, 1.8),
            )
            self.door_poly = poly
            self.shape_type = 99  # tag for logging

        else:
            # ============ original 4 templates (with mirror C) ============
            base = self._polygon_template(self.shape_type)

            # keep your C mirror augmentation if you like
            if self.shape_type == 3 and float(self.np_random.random()) < 0.5:
                base = base.copy()
                base[:, 0] *= -1.0

            sx = float(self.np_random.uniform(0.8, 1.8))
            sz = float(self.np_random.uniform(0.8, 1.8))
            pts_scaled = np.column_stack([sx * base[:, 0], sz * base[:, 1]])
            pts_world = pts_scaled + np.array([door_cx0, door_cz0], dtype=float)

            poly = Polygon(pts_world).buffer(0.0)

            # ✅ NEW: random 360° rotation for templates (data augmentation)
            # rotate around door center (cx0, cz0)
            ang_deg = float(self.np_random.uniform(0.0, 360.0))
            poly = rotate(poly, ang_deg, origin=door_center, use_radians=False).buffer(0.0)

            self.door_poly = poly

        # ----------------------------------------------------------
        # encode polygon -> grid + center + bbox (unchanged)
        # ----------------------------------------------------------
        grid, center, bbox, _ = polygon_to_grid(
            self.door_poly, n=self.grid_n, samples=self.grid_samples, return_meta=True
        )
        self._grid_cached = grid.astype(np.float32)
        self.door_center = center.astype(np.float32)
        self.door_bbox = bbox

        obs = self._build_obs(self._grid_cached, self.door_center, self.theta_prev)
        return obs, {}


    def step(self, action):
        self.step_count += 1

        ax, az, atheta = float(action[0]), float(action[1]), float(action[2])

        xmin, zmin, xmax, zmax = self.door_bbox
        cx0, cz0 = float(self.door_center[0]), float(self.door_center[1])

        # map action to (cx,cz) near center, inside bbox range
        w_half = max((xmax - xmin) / 2.0, 1e-6)
        h_half = max((zmax - zmin) / 2.0, 1e-6)

        cx = cx0 + ax * (0.4 * w_half)
        cz = cz0 + az * (0.4 * h_half)

        theta = atheta * self.theta_max_rad

        # collision check
        corners = self._box_corners_rotated(cx, cz, theta)
        load_poly = Polygon(corners).buffer(0.0)
        all_inside = load_poly.within(self.door_poly)

        # margin (only meaningful when inside; still compute for debug)
        dists = [self.door_poly.exterior.distance(Point(pt)) for pt in corners]
        min_margin = float(min(dists)) if len(dists) > 0 else 0.0

        # ---------------- reward terms (normalized) ----------------
        # center deviation in normalized coordinates
        dx = (cx - cx0) / w_half
        dz = (cz - cz0) / h_half

        # low-z term (keep your original "prefer higher cz" style)
        cz_norm = (cz - zmin) / max((zmax - zmin), 1e-6)

        dtheta = abs(theta - self.theta_prev)

        cost = (
            self.w_pos * (dx * dx + dz * dz)
            + self.w_theta * abs(theta)
            + self.w_dtheta * dtheta
            + self.w_low * cz_norm
        )

        if not all_inside:
            reward = -20.0 - cost
        else:
            reward = 10.0 + self.k_margin * min_margin - cost

        # single-step terminate
        terminated = True
        truncated = False

        # obs: single-step so we can just reuse cached grid/center
        obs = self._build_obs(self._grid_cached, self.door_center, self.theta_prev)

        info = {
            "cx": cx,
            "cz": cz,
            "theta": theta,
            "theta_prev": self.theta_prev,
            "all_inside": all_inside,
            "min_margin": min_margin,
            "shape_type": self.shape_type,
            "door_xy": np.array(self.door_poly.exterior.coords, dtype=float),
            "door_bbox": np.array(self.door_bbox, dtype=float),
        }

        return obs, float(reward), terminated, truncated, info

    # ---------------- helpers ----------------

    def _build_obs(self, grid, center, theta_prev):
        g = np.asarray(grid, dtype=np.float32).reshape(-1)
        c = np.asarray(center, dtype=np.float32).reshape(2,)
        p = np.array([self.L_eff, self.H_eff], dtype=np.float32)
        th_prev_norm = np.array([theta_prev / self.theta_max_rad], dtype=np.float32)
        return np.concatenate([g, c, p, th_prev_norm], axis=0).astype(np.float32)

    def _polygon_template(self, shape_type: int) -> np.ndarray:
        if shape_type == 0:  # rect
            pts = np.array([[-0.25, -0.5],
                            [ 0.25, -0.5],
                            [ 0.25,  0.5],
                            [-0.25,  0.5]], dtype=float)
        elif shape_type == 1:  # tri
            pts = np.array([[-0.3, -0.5],
                            [ 0.3, -0.5],
                            [-0.3,  0.5]], dtype=float)
        elif shape_type == 2:  # pent
            pts = np.array([[-0.25, -0.5],
                            [ 0.25, -0.4],
                            [ 0.35,  0.0],
                            [ 0.10,  0.5],
                            [-0.30,  0.3]], dtype=float)
        else:  # C (concave)
            pts = np.array([[-0.4, -0.6],
                            [ 0.4, -0.6],
                            [ 0.4, -0.3],
                            [ 0.0, -0.3],
                            [ 0.0,  0.3],
                            [ 0.4,  0.3],
                            [ 0.4,  0.6],
                            [-0.4,  0.6]], dtype=float)

        pts -= pts.mean(axis=0)
        return pts

    def _box_corners_rotated(self, cx, cz, th):
        L = self.L_eff
        H = self.H_eff
        base = np.array([[-L/2.0, -H/2.0],
                         [ L/2.0, -H/2.0],
                         [ L/2.0,  H/2.0],
                         [-L/2.0,  H/2.0]], dtype=float)
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]], dtype=float)
        rot = (R @ base.T).T
        return rot + np.array([cx, cz], dtype=float)
