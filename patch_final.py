import re

with open('simulation_demo/animate_rerun.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix 1: Better valid_tris filtering in extrude_poly_to_mesh
old_tris_filter = "valid_tris = [t for t in tris if t.representative_point().within(poly)]"
new_tris_filter = """valid_tris = []
        for t in tris:
            try:
                if t.is_valid and poly.intersection(t).area > 0.99 * t.area:
                    valid_tris.append(t)
            except:
                pass"""
text = text.replace(old_tris_filter, new_tris_filter)

# Fix 2: Buffer 0.05 for hole dilation and specific sizing logic + smaller margin
old_door_logic = '''                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
                    hole_poly = Polygon(poly_xz)
                    
                    # --- 1. 放大可通过区域 (Hole dilation) --- 
                    # 用 buffer(0.15, join_style=2) 放大门洞的尺寸 0.15 米，使视觉上看起来宽敞
                    hole_poly = hole_poly.buffer(0.15, join_style=2)
                    
                    minx, minz, maxx, maxz = hole_poly.bounds
                    
                    # --- 2. 缩小外围墙体尺寸 --- 
                    # 把 1.5m 的 margin 缩减，避免墙体大得夸张
                    margin_x = 0.4
                    margin_z_top = 0.4
                    margin_z_bottom = 0.05
                    wall_poly = Polygon([
                        (minx - margin_x, -margin_z_bottom),
                        (maxx + margin_x, -margin_z_bottom),
                        (maxx + margin_x, maxz + margin_z_top),
                        (minx - margin_x, maxz + margin_z_top)
                    ])
                    
                    # 墙体完整区域扣除门洞的真实多边形
                    wall_with_hole = wall_poly.difference(hole_poly)'''

new_door_logic = '''                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
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
                    wall_poly = Polygon([
                        (minx - margin_x, minz - margin_z_bottom),
                        (maxx + margin_x, minz - margin_z_bottom),
                        (maxx + margin_x, maxz + margin_z_top),
                        (minx - margin_x, maxz + margin_z_top)
                    ])
                    
                    # 用更鲁棒的方式做 difference
                    wall_with_hole = wall_poly.difference(hole_poly).buffer(0)'''

if old_door_logic in text:
    text = text.replace(old_door_logic, new_door_logic)
else:
    print("Door logic not found! (Maybe already replaced?)")

# Fix 3: Colors
text = text.replace('[[60, 120, 220, 200]]', '[[100, 130, 170, 255]]')  # dull blue -> tech grey-blue
text = text.replace('[[60, 120, 220, 220]]', '[[100, 130, 170, 255]]')
text = text.replace('[[60, 120, 220, 255]]', '[[100, 130, 170, 255]]')
text = text.replace('[[220, 60, 60, 255]]', '[[180, 50, 50, 255]]')    # bright red -> slightly muted solid red
text = text.replace('[[220, 60, 60, 220]]', '[[180, 50, 50, 255]]')    # bright red -> slightly muted solid red

with open('simulation_demo/animate_rerun.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patch applied successfully.")
