with open('simulation_demo/animate_rerun.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_section1 = r"""    # ==========================
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    def extrude_poly_to_mesh(poly, ymin, ymax):
        # 使用 shapely 的三角剖分把 2D 多边形转换为 3D Mesh
        tris = triangulate(poly)
        # 过滤掉落在多边形外的三角形
        valid_tris = [t for t in tris if t.representative_point().within(poly)]
        
        vertices = []
        triangles = []
        
        pt_to_idx_front = {}
        pt_to_idx_back = {}
        
        def get_idx(pt, is_front):
            x, z = pt
            d = pt_to_idx_front if is_front else pt_to_idx_back
            y = ymin if is_front else ymax
            key = (round(x, 6), round(z, 6))
            if key not in d:
                d[key] = len(vertices)
                vertices.append([x, y, z])
            return d[key]

        # 前后面三角形
        for t in valid_tris:
            if t.is_empty: continue
            c = list(t.exterior.coords)[:-1]
            if len(c) < 3: continue
            
            i0, i1, i2 = get_idx(c[0], True), get_idx(c[1], True), get_idx(c[2], True)
            # 保证前面和背面的法线朝向对
            triangles.append([i0, i1, i2])
            
            b0, b1, b2 = get_idx(c[0], False), get_idx(c[1], False), get_idx(c[2], False)
            triangles.append([b0, b2, b1])

        # 侧面连接(缝合线)
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            c = list(ring.coords)
            for i in range(len(c)-1):
                p1, p2 = c[i], c[i+1]
                f1, f2 = get_idx(p1, True), get_idx(p2, True)
                b1, b2 = get_idx(p1, False), get_idx(p2, False)
                triangles.append([f1, f2, b1])
                triangles.append([f2, b2, b1])
                
        return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.uint32)

    # 用 Mesh3D 画出完美的 "墙体挖洞" (Hollow Doors - Solid Walls with custom shaped holes)
    if 'doors' in mat:
        doors = mat['doors'][0]
        all_verts = []
        all_tris = []
        idx_offset = 0

        for idx in range(doors.shape[0]):
            try:
                d = doors[idx]
                ymin = float(d['y_min'].item().flatten()[0])
                ymax = float(d['y_max'].item().flatten()[0])
                poly_xz = d['poly_xz']
                
                while hasattr(poly_xz, 'ndim') and poly_xz.ndim == 2 and poly_xz.shape[1] != 2:
                    if poly_xz.shape[0] > 0 and len(poly_xz[0]) > 0:
                        poly_xz = poly_xz[0][0]
                    else: break
                if hasattr(poly_xz, 'item'):
                    try: poly_xz = poly_xz.item()
                    except: pass
                
                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
                    hole_poly = Polygon(poly_xz)
                    # 构造大长方体墙面，再切除 poly_xz 的门洞
                    minx, minz, maxx, maxz = hole_poly.bounds
                    wall_poly = Polygon([
                        (minx - 1.5, 0.0),
                        (maxx + 1.5, 0.0),
                        (maxx + 1.5, maxz + 1.5),
                        (minx - 1.5, maxz + 1.5)
                    ])
                    # 用 difference 扣出门洞
                    wall_with_hole = wall_poly.difference(hole_poly)
                    
                    geoms = getattr(wall_with_hole, "geoms", [wall_with_hole])
                    for g in geoms:
                        if g.is_empty: continue
                        verts, tris = extrude_poly_to_mesh(g, ymin, ymax)
                        if len(verts) > 0:
                            all_verts.extend(verts.tolist())
                            all_tris.extend((tris + idx_offset).tolist())
                            idx_offset += len(verts)
            except Exception as e:
                print(f"Error parsing door {idx}: {e}")
        
        if all_verts:
            rr.log("world/hollow_doors", rr.Mesh3D(
                vertex_positions=all_verts,
                triangle_indices=all_tris,
                vertex_colors=[[60, 120, 220, 200]] * len(all_verts)
            ), static=True)

    # 画实心红色禁飞区 (Red Solid Forbidden)
    if 'forbidden' in mat:
        forbs = mat['forbidden'][0]
        all_verts = []
        all_tris = []
        idx_offset = 0

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
                        verts, tris = extrude_poly_to_mesh(g, ymin, ymax)
                        if len(verts) > 0:
                            all_verts.extend(verts.tolist())
                            all_tris.extend((tris + idx_offset).tolist())
                            idx_offset += len(verts)
            except Exception as e:
                print(f"Error parsing forbidden {idx}: {e}")

        if all_verts:
            rr.log("world/forbidden_solid", rr.Mesh3D(
                vertex_positions=all_verts,
                triangle_indices=all_tris,
                vertex_colors=[[220, 60, 60, 220]] * len(all_verts)
            ), static=True)

    rr.log("world/trajectory_path", rr.LineStrips3D([traj_xyz], colors=[[0, 255, 0, 150]], radii=0.015), static=True)

"""

out = []
in_section1 = False
for line in lines:
    if '# 1. 静态环境渲染 (Static Env)' in line:
        in_section1 = True
        out.append(line)
        out.append(new_section1)
    elif '# 2. 读取并插值载荷的姿态 (Attitude processing)' in line:
        in_section1 = False
        out.append('    # ==========================\n')
        out.append('    # 2. 读取并插值载荷的姿态 (Attitude processing)\n')
    elif not in_section1:
        out.append(line)

with open('simulation_demo/animate_rerun.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
