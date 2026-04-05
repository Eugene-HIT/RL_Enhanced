import re

with open('simulation_demo/animate_rerun.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_extrude = r'''    def extrude_poly_to_mesh(poly, ymin, ymax):
        # 使用 shapely 的三角剖分把 2D 多边形转换为 3D Mesh
        tris = triangulate(poly)
        valid_tris = [t for t in tris if t.representative_point().within(poly)]
        
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
                
        return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.uint32), np.array(normals, dtype=np.float32)'''

start_idx = text.find('    def extrude_poly_to_mesh(poly, ymin, ymax):')
end_idx = text.find('        return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.uint32)')
end_idx = text.find('\n', end_idx)

text = text[:start_idx] + new_extrude + text[end_idx:]

text = text.replace('verts, tris = extrude_poly_to_mesh', 'verts, tris, norms = extrude_poly_to_mesh')

text = text.replace('all_verts.extend(verts.tolist())', 'all_verts.extend(verts.tolist())\n                            all_norms.extend(norms.tolist())')
text = text.replace('f_verts.extend(verts.tolist())', 'f_verts.extend(verts.tolist())\n                            f_norms.extend(norms.tolist())')

text = text.replace('all_verts = []\n        all_tris = []', 'all_verts = []\n        all_tris = []\n        all_norms = []')
text = text.replace('f_verts = []\n        f_tris = []', 'f_verts = []\n        f_tris = []\n        f_norms = []')

text = text.replace('triangle_indices=all_tris,\n                vertex_colors', 'triangle_indices=all_tris,\n                vertex_normals=all_norms,\n                vertex_colors')
text = text.replace('triangle_indices=f_tris,\n                vertex_colors', 'triangle_indices=f_tris,\n                vertex_normals=f_norms,\n                vertex_colors')

with open('simulation_demo/animate_rerun.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated successfully")
