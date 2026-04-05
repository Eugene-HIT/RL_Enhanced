import re

with open('simulation_demo/animate_rerun.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 匹配提取ymin和ymax的行
pattern = r"ymin = float\(d\['y_min'\].item\(\).flatten\(\)\[0\]\)\s*\n\s*ymax = float\(d\['y_max'\].item\(\).flatten\(\)\[0\]\)"

replacement = """ymin_orig = float(d['y_min'].item().flatten()[0])
                ymax_orig = float(d['y_max'].item().flatten()[0])
                # 让障碍物变得更薄（从默认厚度压缩），向中心居中
                target_thickness = 0.25
                y_center = (ymin_orig + ymax_orig) / 2.0
                ymin = y_center - target_thickness / 2.0
                ymax = y_center + target_thickness / 2.0"""

new_text = re.sub(pattern, replacement, text)

with open('simulation_demo/animate_rerun.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Target thickness updated successfully!")
