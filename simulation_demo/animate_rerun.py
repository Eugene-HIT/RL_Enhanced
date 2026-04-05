import rerun as rr
import numpy as np
import scipy.io
import sys
import time
import math

def main():
    print("Initialize Rerun Payload Animation...")
    rr.init("UAV_Payload_Animation", spawn=True)
    
    try:
        mat = scipy.io.loadmat("corridor_export.mat")
        times = np.load("qp3d_sample_t.npy").flatten()
        traj_xyz = np.load("qp3d_sample_xyz.npy")
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    print(f"Loaded trajectory shape: {traj_xyz.shape}, times: {times.shape}")

    # ==========================
    # 1. 静态环境渲染 (Static Env)
    # ==========================
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    def extrude_poly_to_mesh(poly, ymin, ymax):
        # 使用 shapely 的三角剖分把 2D 多边形转换为 3D Mesh
        tris = triangulate(poly)
        valid_tris = []
        for t in tris:
            try:
                if t.is_valid and poly.intersection(t).area > 0.99 * t.area:
                    valid_tris.append(t)
            except:
                pass
        
        vertices = []
        triangles = []
        normals = []
        
        idx = 0
        def add_tri(p0, p1, p2, n):
            nonlocal idx
            vertices.extend([p0, p1, p2])
            normals.extend([n, n, n])
            triangles.append([idx, idx+1, idx+2])
            idx += 3

        def get_normal(p0, p1, p2):
            u = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]
            v = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]]
            nx = u[1]*v[2] - u[2]*v[1]
            ny = u[2]*v[0] - u[0]*v[2]
            nz = u[0]*v[1] - u[1]*v[0]
            L = math.hypot(math.hypot(nx, ny), nz)
            if L > 1e-6:
                return [nx/L, ny/L, nz/L]
            return [0, 1, 0]

        # 前后面
        for t in valid_tris:
            if t.is_empty: continue
            c = list(t.exterior.coords)[:-1]
            if len(c) < 3: continue
            
            p0 = [c[0][0], ymin, c[0][1]]
            p1 = [c[1][0], ymin, c[1][1]]
            p2 = [c[2][0], ymin, c[2][1]]
            n = get_normal(p0, p1, p2)
            if n[1] > 0: # 保证法线朝外(-y)
                p1, p2 = p2, p1
                n = get_normal(p0, p1, p2)
            add_tri(p0, p1, p2, n)
            
            p0b = [c[0][0], ymax, c[0][1]]
            p1b = [c[1][0], ymax, c[1][1]]
            p2b = [c[2][0], ymax, c[2][1]]
            nb = get_normal(p0b, p1b, p2b)
            if nb[1] < 0: # 保证法线朝外(+y)
                p1b, p2b = p2b, p1b
                nb = get_normal(p0b, p1b, p2b)
            add_tri(p0b, p1b, p2b, nb)

        # 侧面(缝合线)
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            c = list(ring.coords)
            for i in range(len(c)-1):
                p1_xz = c[i]
                p2_xz = c[i+1]
                
                # 侧面也是需要根据绕向计算正确的法线
                # P0 ---- P1 (ymin)
                # |        |
                # P2 ---- P3 (ymax)
                p0 = [p1_xz[0], ymin, p1_xz[1]]
                p1 = [p2_xz[0], ymin, p2_xz[1]]
                p2 = [p1_xz[0], ymax, p1_xz[1]]
                p3 = [p2_xz[0], ymax, p2_xz[1]]
                
                # quad 拆分成两个三角形
                # tri 1: p0, p2, p3
                n1 = get_normal(p0, p2, p3)
                add_tri(p0, p2, p3, n1)
                # tri 2: p0, p3, p1
                n2 = get_normal(p0, p3, p1)
                add_tri(p0, p3, p1, n2)
                
        return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.uint32), np.array(normals, dtype=np.float32)

    # 用 Mesh3D 画出完美的 "大长方体挖去门洞的形状" 
    if 'doors' in mat:
        doors = mat['doors'][0]
        # 给所有的门同一个 Mesh
        all_verts = []
        all_tris = []
        all_norms = []
        idx_offset = 0

        for idx in range(doors.shape[0]):
            try:
                d = doors[idx]
                # 完全保留原本的长方形厚度
                ymin_orig = float(d['y_min'].item().flatten()[0])
                ymax_orig = float(d['y_max'].item().flatten()[0])
                # 让障碍物变得更薄（从默认厚度压缩），向中心居中
                target_thickness = 0.6
                y_center = (ymin_orig + ymax_orig) / 2.0
                ymin = y_center - target_thickness / 2.0
                ymax = y_center + target_thickness / 2.0
                poly_xz = d['poly_xz']
                
                while hasattr(poly_xz, 'ndim') and poly_xz.ndim == 2 and poly_xz.shape[1] != 2:
                    if poly_xz.shape[0] > 0 and len(poly_xz[0]) > 0:
                        poly_xz = poly_xz[0][0]
                    else: break
                if hasattr(poly_xz, 'item'):
                    try: poly_xz = poly_xz.item()
                    except: pass
                
                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
                    # 先加上 buffer(0) 修复可能的自交
                    hole_poly = Polygon(poly_xz).buffer(0)
                    
                    # C字形连线问题常常是因为内凹多边形 buffer 时产生自交，
                    # 最后一个门洞(idx == 3)需要小一点。
                    if idx == 3:  # 最后一个长方形门洞
                        hole_poly = hole_poly.buffer(0.0)
                    else:
                        hole_poly = hole_poly.buffer(0.15, join_style=2).buffer(0)
                    
                    minx, minz, maxx, maxz = hole_poly.bounds
                    
                    # 再次大幅度减小外围尺寸，使得它贴合
                    margin_x = 0.2
                    margin_z_top = 0.15
                    margin_z_bottom = 0.02
                    
                    if idx == 2:
                        # 让右侧梯形斜边与内部直角三角形的斜边严格数学平行：
                        # 内部斜率 dx=0.72, dz=-1.55，计算对应的顶部偏置
                        dx_slope = 0.72
                        dz_slope = 1.55
                        bottom_right_x = maxx + margin_x + 0.2
                        total_h = (maxz + margin_z_top) - (-0.05)
                        top_right_x = bottom_right_x - total_h * (dx_slope / dz_slope)
                        
                        wall_poly = Polygon([
                            (minx - margin_x, -0.05),
                            (bottom_right_x, -0.05),
                            (top_right_x, maxz + margin_z_top),
                            (minx - margin_x, maxz + margin_z_top)
                        ])
                    else:
                        wall_poly = Polygon([
                            (minx - margin_x, minz - margin_z_bottom),
                            (maxx + margin_x, minz - margin_z_bottom),
                            (maxx + margin_x, maxz + margin_z_top),
                            (minx - margin_x, maxz + margin_z_top)
                        ])
                    
                    # 用更鲁棒的方式做 difference
                    wall_with_hole = wall_poly.difference(hole_poly).buffer(0)
                    
                    geoms = getattr(wall_with_hole, "geoms", [wall_with_hole])
                    for g in geoms:
                        if g.is_empty: continue
                        verts, tris, norms = extrude_poly_to_mesh(g, ymin, ymax)
                        if len(verts) > 0:
                            all_verts.extend(verts.tolist())
                            all_norms.extend(norms.tolist())
                            all_tris.extend((tris + idx_offset).tolist())
                            idx_offset += len(verts)
            except Exception as e:
                print(f"Error parsing door {idx}: {e}")
        
        if all_verts:
            rr.log("world/hollow_doors", rr.Mesh3D(
                vertex_positions=all_verts,
                triangle_indices=all_tris,
                vertex_normals=all_norms,
                vertex_colors=[[100, 130, 170, 140]] * len(all_verts)
            ), static=True)

    # 包含红色实心不可穿透障碍 (Red Solid Forbidden)
    if 'forbidden' in mat:
        forbs = mat['forbidden'][0]
        # 用 Mesh 绘制，支持任何形状的不可穿透障碍
        f_verts = []
        f_tris = []
        f_norms = []
        f_offset = 0

        for idx in range(forbs.shape[0]):
            try:
                f = forbs[idx]
                ymin = float(f['y_min'].item().flatten()[0])
                ymax = float(f['y_max'].item().flatten()[0])
                poly_xz = f['poly_xz']
                
                while hasattr(poly_xz, 'ndim') and poly_xz.ndim == 2 and poly_xz.shape[1] != 2:
                    if poly_xz.shape[0] > 0 and len(poly_xz[0]) > 0:
                        poly_xz = poly_xz[0][0]
                    else: break
                if hasattr(poly_xz, 'item'):
                    try: poly_xz = poly_xz.item()
                    except: pass
                
                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
                    f_poly = Polygon(poly_xz)
                    geoms = getattr(f_poly, "geoms", [f_poly])
                    for g in geoms:
                        if g.is_empty: continue
                        verts, tris, norms = extrude_poly_to_mesh(g, ymin, ymax)
                        if len(verts) > 0:
                            f_verts.extend(verts.tolist())
                            f_norms.extend(norms.tolist())
                            f_tris.extend((tris + f_offset).tolist())
                            f_offset += len(verts)
            except Exception as e:
                pass

        if f_verts:
            rr.log("world/forbidden_solid", rr.Mesh3D(
                vertex_positions=f_verts,
                triangle_indices=f_tris,
                vertex_normals=f_norms,
                vertex_colors=[[180, 50, 50, 120]] * len(f_verts)
            ), static=True)

    rr.log("world/trajectory_path", rr.LineStrips3D([traj_xyz], colors=[[0, 255, 0, 150]], radii=0.015), static=True)

    # ==========================
    # 2. 读取并插值载荷的姿态 (Attitude processing)
    # ==========================
    if 'keyframes' in mat and 'traj' in mat:
        roll_wp = mat['keyframes'][0][0]['roll_wp'].flatten()
        T_per_seg = mat['traj'][0][0]['T_per_seg'].flatten()
        
        # 构建时间轴映射，用来把控制点的姿态铺满整个连续轨迹
        wp_t = np.zeros(len(T_per_seg) + 1)
        wp_t[1:] = np.cumsum(T_per_seg)
        
        # 以时间为基准进行线性插值，推算出每一帧动画对应的倾斜角
        roll_t = np.interp(times, wp_t, roll_wp)
    else:
        print("Warning: Couldn't find roll_wp, using zero attitude.")
        roll_t = np.zeros(len(times))


    # ==========================
    # 3. 按时间注入动画帧 (Animation)
    # ==========================
    print("Generating animation sequence...")
    
    from scipy.spatial.transform import Rotation as R
    
    # 预先定义载荷的尺寸 (L=0.6, W/Y=0.4, H=0.2)
    payload_half_sizes = [[0.3, 0.2, 0.1]]
    payload_colors = [[240, 100, 0, 255]] # 亮橙色表示有效载荷
    
    # --- 构建更真实的三机协同悬吊系统 ---
    # 缩小化的无人机尺寸
    uav_body_half_size = [0.08, 0.08, 0.02]
    uav_color = [[50, 50, 50, 255]]
    rotor_offsets = [
        [0.1, 0.1, 0.01], [0.1, -0.1, 0.01],
        [-0.1, 0.1, 0.01], [-0.1, -0.1, 0.01]
    ]
    uav_part_centers = [[0, 0, 0]] + rotor_offsets
    uav_part_halfs = [uav_body_half_size] + [[0.05, 0.05, 0.005]] * 4
    uav_part_colors = uav_color + [[150, 200, 250, 180]] * 4
    
    # 先将载荷本体和三个无人机的静态结构注册好(static=True能够提升性能)
    rr.log("world/payload_system/payload_box", rr.Boxes3D(centers=[[0,0,0]], half_sizes=payload_half_sizes, colors=payload_colors, fill_mode=rr.components.FillMode.Solid), static=True)
    for j in range(3):
        rr.log(f"world/payload_system/uav_{j}_boxes", rr.Boxes3D(centers=uav_part_centers, half_sizes=uav_part_halfs, colors=uav_part_colors, fill_mode=rr.components.FillMode.Solid), static=True)

    # 三个无人机相对于载荷中心投影的全局固定高度和间距编队 (构成一前两后的稳定三角形)
    # dx, dy, dz - 假设无人机在全局系下一直保持这个编队队形
    uav_global_offsets = [
        [ 0.25,  0.0,  0.45],  # UAV 0: 正前方，高度 0.45m
        [-0.25,  0.30, 0.45],  # UAV 1: 左后方，高度 0.45m
        [-0.25, -0.30, 0.45]   # UAV 2: 右后方，高度 0.45m
    ]

    # 三根悬吊绳分别连接载荷上的三个固定挂载点 (相对于载荷局部坐标系)
    payload_attach_locals = [
        [ 0.25,  0.0,  0.1],  # 对应 UAV 0的挂点 (载荷顶部前部)
        [-0.25,  0.18, 0.1],  # 对应 UAV 1的挂点 (载荷顶部左后)
        [-0.25, -0.18, 0.1]   # 对应 UAV 2的挂点 (载荷顶部右后)
    ]

    # 将载荷系统按时间驱动
    for i in range(len(times)):
        t = float(times[i])
        p = traj_xyz[i]
        
        # RL网络规划出来的载荷在 XZ 面内的倾斜角 (Rotation around Y-axis)
        theta = float(roll_t[i])
        
        # 告诉 Rerun 要更新动画时刻
        rr.set_time_seconds("play_time", t) if hasattr(rr, "set_time_seconds") else rr.set_time("play_time", duration=t)
        
        # --- 旋转载荷本体 ---
        rr.log(
            "world/payload_system/payload_box", 
            rr.Transform3D(
                translation=p,
                rotation=rr.RotationAxisAngle(axis=[0, 1, 0], radians=-theta)
            )
        )
        
        # --- 利用空间旋转计算精确的三机挂载位置 ---
        # 注意此前的偏角定义：模型里绕Y轴转，向后仰也就是pitch。这里根据你之前的结果取的是-theta
        rot = R.from_rotvec([0, -theta, 0])
        
        cable_lines = []
        for j in range(3):
            # 无人机在全局坐标系中，位于载荷平移基准上方附近的编队位置
            p_uav = [p[0] + uav_global_offsets[j][0],
                     p[1] + uav_global_offsets[j][1],
                     p[2] + uav_global_offsets[j][2]]

            # 计算局部挂载点随载荷旋转后，在全局坐标系下的绝对位置
            # p_attach_global = 载荷中心全区坐标(p) + 经过旋转后的局部偏置点
            p_attach_global = p + rot.apply(payload_attach_locals[j])

            # 将这根牵引绳加入到渲染列表中
            cable_lines.append([p_attach_global.tolist(), p_uav])

            # 在保持无人机自身水平不旋转的情况下，移动无人机模型到对应全局位置
            rr.log(f"world/payload_system/uav_{j}_boxes", rr.Transform3D(translation=p_uav))
            
        # 每一帧绘制三根动态牵引绳，精准连接无人机中心和倾斜中载荷的三个角点
        rr.log("world/payload_system/cables", rr.LineStrips3D(cable_lines, colors=[[30, 30, 30, 255]]*3, radii=0.003))

    print("\n>>> Done! Check Rerun Viewer.")
    print(">>> 【重要提示】：请在界面的最下方时间轴，将时间线切换到 'play_time' 面板。")
    print(">>> 然后点击左下方的播放按钮 (▶️)，即可查看规划好的载荷姿态动画！")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
