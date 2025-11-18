import json
import numpy as np

# 把你 README 里的数值粘到这里
qw = 0.6611149473637592
qx = 0.010386763815325842
qy = -0.7499722379641203
qz = -0.01899430948330976
tx = -0.07988991299367104
ty = 0.01797496318085282
tz = 0.01734501621123293

def quat_to_R(qw,qx,qy,qz):
    # 归一化
    n = (qw**2+qx**2+qy**2+qz**2)**0.5
    qw,qx,qy,qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    return R

T = np.eye(4, dtype=float)
T[:3,:3] = quat_to_R(qw,qx,qy,qz)
T[:3,  3] = [tx,ty,tz]

# 假设：该 T 就是 T_ee_cam（末端->相机）。若经验证发现相反（T_cam_ee），用 np.linalg.inv(T)。
T_ee_cam = T  
# 或者：np.linalg.inv(T)

with open("hand_eye.json", "w", encoding="utf-8") as f:
    json.dump({"T_ee_cam": T_ee_cam.tolist()}, f, indent=2)
print("Saved hand_eye.json")