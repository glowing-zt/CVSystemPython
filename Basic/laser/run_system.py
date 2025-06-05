
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ocam_model import get_virtual_ocam_model
from las_segm import las_segm
from mapping import mapping
from mock_cube_dist import mock_cube_dist  # 简化深度估计方法
from laser_debug import print_laser_bounds

# === 配置参数 ===
image_path = "D:/Basic/laser/image.jpg"
x_angle = 1.5
y_angle = -2.5
las_dist = 950
CVsyst_x = 0
CVsyst_y = 0

# === 加载图像和模型 ===
img_bin = las_segm(image_path)
print_laser_bounds(img_bin)

ocam_model = get_virtual_ocam_model()

# === 计算整体映射点（可视化） ===
x1, y1 = mapping(img_bin, x_angle, y_angle, las_dist, ocam_model)

# === 模拟深度估计 ===
C_left = mock_cube_dist(img_bin, [890, 940], [350, 600], name="Left Cube")
C_Up = mock_cube_dist(img_bin, [250, 300], [700, 950], name="Up Cube")
C_Right = mock_cube_dist(img_bin, [500, 570], [1200, 1500], name="Right Cube")

# === 打印估计结果 ===
print(f"Left Cube Distance Estimate: {C_left:.2f} mm")
print(f"Up Cube Distance Estimate: {C_Up:.2f} mm")
print(f"Right Cube Distance Estimate: {C_Right:.2f} mm")

# === 可视化图像（保存不显示）===
plt.figure(figsize=(8, 6))
plt.scatter(x1, y1, s=3, c='b', label='Laser Intersections')
plt.scatter([CVsyst_x], [-CVsyst_y], c='r', marker='*', s=100, label='Camera')
plt.title("3D Mapping of Laser Line Intersections")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mapping_result.png")
plt.close()

# === 误差分析（基于参考真实值） ===
real_left = -922.00
real_up = 1799.00
real_right = 521.00

err_left = abs(real_left - C_left)
err_up = abs(real_up - C_Up)
err_right = abs(real_right - C_Right)

print("\n====== Error Analysis Table ======")
print(f"Left Cube  | Real: {real_left:.2f} mm | Predicted: {C_left:.2f} mm | Error: {err_left:.2f} mm")
print(f"Up Cube    | Real: {real_up:.2f} mm  | Predicted: {C_Up:.2f} mm  | Error: {err_up:.2f} mm")
print(f"Right Cube | Real: {real_right:.2f} mm | Predicted: {C_Right:.2f} mm | Error: {err_right:.2f} mm")

cv2.imwrite("debug_mask.png", img_bin * 255)
