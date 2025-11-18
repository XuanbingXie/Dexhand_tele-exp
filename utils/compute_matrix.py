import json
import math

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Compute rotation matrix elements
    r11 = 1 - 2*(qy**2 + qz**2)
    r12 = 2*(qx*qy - qz*qw)
    r13 = 2*(qx*qz + qy*qw)
    
    r21 = 2*(qx*qy + qz*qw)
    r22 = 1 - 2*(qx**2 + qz**2)
    r23 = 2*(qy*qz - qx*qw)
    
    r31 = 2*(qx*qz - qy*qw)
    r32 = 2*(qy*qz + qx*qw)
    r33 = 1 - 2*(qx**2 + qy**2)
    
    return [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]

# 新的手眼标定数据（OpenCV/Tsai-Lenz方法）
tx = -0.07988991299367104
ty = 0.01797496318085282
tz = 0.01734501621123293

qx = 0.010386763815325842
qy = -0.7499722379641203
qz = -0.01899430948330976
qw = 0.6611149473637592

# 转换为旋转矩阵
R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

# 构建4x4变换矩阵 T_ee_cam (Link6 -> Camera)
T_ee_cam = [
    [R[0][0], R[0][1], R[0][2], tx],
    [R[1][0], R[1][1], R[1][2], ty],
    [R[2][0], R[2][1], R[2][2], tz],
    [0.0, 0.0, 0.0, 1.0]
]

# 工具坐标系偏移 (Link6 -> Gripper, Z轴+180mm)
T_link6_gripper = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.18],
    [0.0, 0.0, 0.0, 1.0]
]

# 保存为JSON
data = {
    "T_ee_cam": T_ee_cam,
    "T_link6_gripper": T_link6_gripper
}

with open("hand_eye.json", "w") as f:
    json.dump(data, f, indent=2)

print("hand_eye.json has been updated with new calibration!")
print("\nNew T_ee_cam matrix:")
for row in T_ee_cam:
    print(f"  {row}")

print("\nOld vs New calibration differences:")
print(f"  Delta_X = {-0.073 - tx:.4f}m = {(-0.073 - tx)*1000:.1f}mm")
print(f"  Delta_Y = {0.019 - ty:.4f}m = {(0.019 - ty)*1000:.1f}mm")
print(f"  Delta_Z = {0.022 - tz:.4f}m = {(0.022 - tz)*1000:.1f}mm")
print("\nThese differences caused grasp position offset. Should be fixed now!")
