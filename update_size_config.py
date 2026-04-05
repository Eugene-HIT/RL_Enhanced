import re

with open('simulation_demo/animate_rerun.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_str = '''                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
                    hole_poly = Polygon(poly_xz)
                    minx, minz, maxx, maxz = hole_poly.bounds
                    
                    # 构造包含该门洞的大长方体墙面（稍微扩大范围模拟大型障碍墙）
                    wall_poly = Polygon([
                        (minx - 1.5, -0.05),
                        (maxx + 1.5, -0.05),
                        (maxx + 1.5, maxz + 1.5),
                        (minx - 1.5, maxz + 1.5)
                    ])
                    # 墙体完整区域扣除门洞的真实多边形
                    wall_with_hole = wall_poly.difference(hole_poly)'''

new_str = '''                if isinstance(poly_xz, np.ndarray) and poly_xz.ndim == 2 and poly_xz.shape[1] == 2:
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

if old_str in text:
    print('Pattern match! Replacing...')
    text = text.replace(old_str, new_str)
    with open('simulation_demo/animate_rerun.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Replaced successfully.')
else:
    print('Pattern not found!')
