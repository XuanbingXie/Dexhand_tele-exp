import numpy as np
import json

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

# 新的手眼标定数据
tx = -0.07988991299367104
ty = 0.01797496318085282
tz = 0.01734501621123293

qx = 0.010386763815325842
qy = -0.7499722379641203
qz = -0.01899430948330976
qw = 0.6611149473637592

# 转换为旋转矩阵
R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

# 构建4x4变换矩阵
T_ee_cam = np.eye(4)
T_ee_cam[:3, :3] = R
T_ee_cam[:3, 3] = [tx, ty, tz]

print("新的 T_ee_cam (Link6 -> Camera):")
print(T_ee_cam)
print("\n")

# 工具坐标系偏移 (Link6 -> Gripper)
T_link6_gripper = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.18],  # Z轴180mm偏移
    [0.0, 0.0, 0.0, 1.0]
])

# 保存为JSON格式
data = {
    "T_ee_cam": T_ee_cam.tolist(),
    "T_link6_gripper": T_link6_gripper.tolist()
}

with open("hand_eye.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ hand_eye.json 已更新！")
print("\n差异分析:")
print(f"平移差异: ΔX={-0.073 - tx:.4f}m, ΔY={0.019 - ty:.4f}m, ΔZ={0.022 - tz:.4f}m")
