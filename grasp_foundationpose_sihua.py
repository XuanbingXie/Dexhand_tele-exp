import os
import sys
import json
import time
import argparse
from typing import Tuple
import numpy as np

# Make local packages importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "FoundationPose"))

from FoundationPose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis  # type: ignore
from FoundationPose.offscreen_renderer import dr  # type: ignore
import trimesh  # type: ignore
import pyrealsense2 as rs  # type: ignore
import cv2  # type: ignore

from RM_API2.RealMan.realman import RobotArmController  # robot API wrapper
from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi  # gripper


def load_hand_eye_transform(hand_eye_path: str) -> np.ndarray:
    with open(hand_eye_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    T = np.asarray(data["T_ee_cam"], dtype=np.float64)
    assert T.shape == (4, 4), "T_ee_cam must be 4x4"
    return T


def load_tool_transform(hand_eye_path: str) -> np.ndarray:
    """
    Load tool coordinate system offset (Link6 -> Gripper/TCP).
    If not defined in hand_eye.json, returns identity matrix.
    """
    with open(hand_eye_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "T_link6_gripper" in data:
        T = np.asarray(data["T_link6_gripper"], dtype=np.float64)
        assert T.shape == (4, 4), "T_link6_gripper must be 4x4"
        return T
    else:
        # No tool offset defined, return identity (assume get_current_pose returns Link6)
        return np.eye(4, dtype=np.float64)


def rmat_to_rvec_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to ZYX intrinsic Euler angles (rx, ry, rz) in radians.
    Returns (rx, ry, rz) corresponding to rotations about X, Y, Z respectively.
    """
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)
    if abs(np.cos(ry)) < 1e-6:
        rx = np.arctan2(-R[0, 1], R[1, 1])
        rz = 0.0
    else:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    return float(rx), float(ry), float(rz)


def get_mask_from_user(color: np.ndarray, depth: np.ndarray) -> np.ndarray:
    roi = cv2.selectROI("Select Object (ENTER)", color[..., ::-1], False, False)
    cv2.destroyWindow("Select Object (ENTER)")
    mask = np.zeros((color.shape[0], color.shape[1]), dtype=bool)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return mask

    mask[y : y + h, x : x + w] = True

    depth_roi = depth[y : y + h, x : x + w]
    valid_depth = depth_roi[depth_roi > 0]
    if len(valid_depth) > 0:
        depth_mean = np.median(valid_depth)
        depth_threshold = 0.08  # 原来 0.15，缩小以更好区分物体与桌面
        depth_mask = (np.abs(depth - depth_mean) < depth_threshold) & (depth > 0)
    else:
        depth_mask = depth > 0

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray_roi = gray[y : y + h, x : x + w]

    roi_mean = float(np.mean(gray_roi))
    roi_std = float(np.std(gray_roi))

    k = 0.8
    intensity_thresh = roi_mean - k * roi_std
    intensity_thresh = max(20.0, min(220.0, intensity_thresh))

    intensity_mask = gray < intensity_thresh

    combined = mask & depth_mask & intensity_mask

    if combined.sum() < 50:
        combined = mask & depth_mask

    return combined


def detect_pose_with_foundationpose(mesh_file: str, debug_dir: str = "fp_debug", use_centroid: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Returns:
      pose_cam_obj (4x4): object pose in camera frame (T_cam_obj)
      calib: dict with K and other info
      
    Args:
      use_centroid: If True, use mesh centroid as grasp point instead of OBB center
    """
    code_dir = os.path.join(ROOT_DIR, "FoundationPose")
    os.makedirs(debug_dir, exist_ok=True)

    mesh = trimesh.load(mesh_file)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=1,
        glctx=glctx,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    try:
        # Registration
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()) * depth_scale

            mask = get_mask_from_user(color, depth)
            if mask.sum() == 0:
                continue
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
            if pose is not None:
                # Calculate grasp center
                if use_centroid:
                    # Use mesh centroid (center of mass)
                    center_offset = mesh.centroid
                    mesh_to_center = np.eye(4, dtype=np.float64)
                    mesh_to_center[:3, 3] = center_offset
                else:
                    # Use OBB center (original behavior)
                    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                    mesh_to_center = np.linalg.inv(to_origin)
                
                # Visualization
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
                center_pose = pose @ mesh_to_center
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow("FoundationPose", vis[..., ::-1])
                cv2.waitKey(500)
                cv2.destroyAllWindows()
                return pose, {
                    "K": K,
                    "depth_scale": depth_scale,
                    "mesh_to_center": mesh_to_center,
                }
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def homogenize(p: np.ndarray) -> np.ndarray:
    assert p.shape == (3,)
    h = np.eye(4, dtype=np.float64)
    h[:3, 3] = p
    return h


def make_translation(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [dx, dy, dz]
    return T


def get_camera_to_rm_transform() -> np.ndarray:
    """
    Convert from camera coordinate system (OpenCV/RealSense) to RM base coordinate system.
    
    Camera frame (OpenCV/RealSense): X右, Y下, Z前
    RM base frame: X前, Y左, Z上
    
    Transformation:
    - RM_X = Camera_Z (前)
    - RM_Y = -Camera_X (左，相机X是右，所以取负)
    - RM_Z = -Camera_Y (上，相机Y是下，所以取负)
    
    Returns:
        T_rm_cam (4x4): Transformation from camera frame to RM base frame
    """
    # Rotation matrix: [RM_X, RM_Y, RM_Z] = [Cam_Z, -Cam_X, -Cam_Y]
    R_rm_cam = np.array([
        [0,  0,  1],   # RM_X = Camera_Z
        [-1, 0,  0],   # RM_Y = -Camera_X
        [0, -1,  0],   # RM_Z = -Camera_Y
    ], dtype=np.float64)
    
    T_rm_cam = np.eye(4, dtype=np.float64)
    T_rm_cam[:3, :3] = R_rm_cam
    return T_rm_cam


def interpolate_pose(pose_start: list, pose_end: list, alpha: float) -> list:
    """
    线性插值两个位姿，用于生成平滑路径点
    
    Args:
        pose_start: 起始位姿 [x, y, z, rx, ry, rz]
        pose_end: 目标位姿 [x, y, z, rx, ry, rz]
        alpha: 插值系数 [0, 1]，0=起点，1=终点
    
    Returns:
        插值后的位姿
    """
    pose = []
    for i in range(6):
        pose.append((1 - alpha) * pose_start[i] + alpha * pose_end[i])
    return pose


def generate_waypoints(pose_start: list, pose_end: list, num_points: int = 5) -> list:
    """
    在两个位姿之间生成多个中间路径点
    
    Args:
        pose_start: 起始位姿
        pose_end: 目标位姿
        num_points: 中间点数量（不包括起点和终点）
    
    Returns:
        包含所有路径点的列表
    """
    waypoints = [pose_start]
    for i in range(1, num_points + 1):
        alpha = i / (num_points + 1)
        waypoint = interpolate_pose(pose_start, pose_end, alpha)
        waypoints.append(waypoint)
    waypoints.append(pose_end)
    return waypoints


def move_smooth(robot: RobotArmController, target_pose: list, 
                current_pose: list = None, 
                velocity: float = 10, 
                num_waypoints: int = 3) -> bool:
    """
    平滑移动到目标位姿，通过插入中间路径点避免关节突变
    警告：只使用 movel (笛卡尔空间运动)，不要用 movej！
    
    Args:
        robot: 机械臂控制器
        target_pose: 目标位姿 [x, y, z, rx, ry, rz] (笛卡尔坐标)
        current_pose: 当前位姿（如果为None则自动获取）
        velocity: 运动速度 (mm/s)
        num_waypoints: 中间路径点数量
    
    Returns:
        是否成功
    """
    if current_pose is None:
        current_pose = list(robot.get_current_pose())
    
    # 检查是否需要插入路径点（距离较远时）
    pos_diff = np.linalg.norm(np.array(target_pose[:3]) - np.array(current_pose[:3]))
    angle_diff = np.linalg.norm(np.array(target_pose[3:]) - np.array(current_pose[3:]))
    
    print(f"Position difference: {pos_diff:.3f}m, Angle difference: {angle_diff:.3f}rad")
    
    # 如果距离很小，直接移动
    if pos_diff < 0.05 and angle_diff < 0.3:
        print("Small motion, direct movel")
        try:
            robot.movel(target_pose, v=velocity)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Error: Failed to move: {e}")
            return False
    
    # 距离较大，生成路径点
    print(f"Generating {num_waypoints} waypoints for smooth motion...")
    waypoints = generate_waypoints(current_pose, target_pose, num_waypoints)
    
    # 逐个移动到路径点（只用 movel）
    for i, waypoint in enumerate(waypoints[1:], 1):  # 跳过起点
        print(f"Moving to waypoint {i}/{len(waypoints)-1}")
        try:
            robot.movel(waypoint, v=velocity)
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error: Failed to reach waypoint {i}: {e}")
            print("Aborting smooth motion for safety")
            return False
    
    print("Smooth motion completed")
    return True


def move_to_safe_height(robot: RobotArmController, safe_z: float = 0.2, velocity: float = 20) -> bool:
    """
    移动到安全高度（避免碰撞）
    只使用 movel，垂直上升
    
    Args:
        robot: 机械臂控制器
        safe_z: 安全高度（米）
        velocity: 运动速度 (mm/s)
    
    Returns:
        是否成功
    """
    current_pose = list(robot.get_current_pose())
    
    # 只改变Z高度，保持其他不变
    if current_pose[2] < safe_z:
        print(f"Lifting to safe height: {safe_z}m")
        safe_pose = current_pose.copy()
        safe_pose[2] = safe_z
        
        try:
            robot.movel(safe_pose, v=velocity)
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"Error: Failed to reach safe height: {e}")
            return False
    else:
        print(f"Already at safe height: {current_pose[2]:.3f}m")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--hand_eye", type=str, required=True, help="JSON file containing {'T_ee_cam': [[...],[...],[...],[...]]}")
    parser.add_argument("--pregrasp_height", type=float, default=0.01, help="pre-grasp height above object in RM Z direction (m)")
    parser.add_argument("--grasp_offset", type=float, default=0.0, help="final Z offset from object position for grasp (m)")
    parser.add_argument("--hand_speed", type=int, default=120)
    
    # 运动平滑参数
    parser.add_argument("--safe_height", type=float, default=0.2, help="Safe height for transit motion (m)")
    parser.add_argument("--approach_velocity", type=float, default=10, help="Velocity for approaching object (mm/s)")
    parser.add_argument("--transit_velocity", type=float, default=20, help="Velocity for transit motion (mm/s)")
    parser.add_argument("--num_waypoints", type=int, default=10, help="Number of waypoints for smooth motion")
    
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    robot = RobotArmController("192.168.1.28", 8080, 3)
    hand = LinkerHandApi(hand_type="right", hand_joint="L10")
    hand.set_speed(speed=[120,200,200,200,200])

    observation_point = [0.473533,-0.145475,-0.028986,-3.058,-0.717,0.629]
    current = list(robot.get_current_pose())
    
    # 使用平滑运动到观察位置
    if not move_smooth(robot, observation_point, current, 
                       velocity=args.transit_velocity, 
                       num_waypoints=args.num_waypoints):
        print("Failed to reach observation pose")
        robot.disconnect()
        return
    
    hand.finger_move([255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
    time.sleep(0.5)

    pose_cam_obj, info = detect_pose_with_foundationpose(args.mesh_file, debug_dir=os.path.join(ROOT_DIR, "fp_debug"))
    mesh_to_center = info.get("mesh_to_center", np.eye(4, dtype=np.float64))
    T_cam_obj = pose_cam_obj @ mesh_to_center

    current_pose = robot.get_current_pose()
    x, y, z, rx, ry, rz = current_pose

    # Build T_base_tcp from current pose (assume rx,ry,rz is ZYX euler in radians)
    # Note: If tool coordinate is set, this is T_base_gripper; otherwise T_base_link6
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    R_base_tcp = Rz @ Ry @ Rx
    T_base_tcp = np.eye(4, dtype=np.float64)
    T_base_tcp[:3, :3] = R_base_tcp
    T_base_tcp[:3, 3] = [x, y, z]

    # Load hand-eye calibration (Link6 -> Camera)
    T_link6_cam = load_hand_eye_transform(args.hand_eye)
    
    # Load tool offset (Link6 -> Gripper), if tool coordinate is set
    T_link6_gripper = load_tool_transform(args.hand_eye)
    
    # Compute T_base_link6
    # If tool coordinate is NOT set: T_base_tcp = T_base_link6 (identity compensation)
    # If tool coordinate IS set: T_base_tcp = T_base_gripper, need to compensate
    T_gripper_link6 = np.linalg.inv(T_link6_gripper)
    T_base_link6 = T_base_tcp @ T_gripper_link6
    
    # Compute object pose in base frame with coordinate system conversion
    # 由于手眼标定是在 OpenCV 坐标系下做的，需要转换到 RM 坐标系
    # OpenCV: X右, Y下, Z前  →  RM: X前, Y左, Z上
    T_rm_cam = get_camera_to_rm_transform()
    T_cam_obj_rm = T_rm_cam @ T_cam_obj
    T_link6_cam_rm = T_rm_cam @ T_link6_cam
    T_cam_link6_rm = np.linalg.inv(T_link6_cam_rm)
    T_link6_obj = T_cam_link6_rm @ T_cam_obj_rm
    T_base_obj = T_base_link6 @ T_link6_obj

    # 5) Define grasp strategy: move to pre-grasp above object, then to grasp position
    R_base_tool = R_base_tcp
    p_obj = T_base_obj[:3, 3]

    # RM坐标系：X前，Y左，Z上
    p_pregrasp = p_obj + np.array([0, 0, args.pregrasp_height])  # 沿Z轴向上
    p_grasp = p_obj + np.array([0, 0, args.grasp_offset])  # 可以沿Z轴微调

    rx_cmd, ry_cmd, rz_cmd = rmat_to_rvec_zyx(R_base_tool)
    pose_pregrasp = [float(p_pregrasp[0]), float(p_pregrasp[1]), -0.14, rx_cmd, ry_cmd, rz_cmd]
    pose_grasp = [float(p_grasp[0]), float(p_grasp[1]), float(p_grasp[2]), rx_cmd, ry_cmd, rz_cmd]

    print(f"Pre-grasp pose: {pose_pregrasp}")
    import pdb; pdb.set_trace()
    if not move_to_safe_height(robot, safe_z=args.safe_height, velocity=args.transit_velocity):
        print("Failed to reach safe height, aborting...")
        robot.disconnect()
        return
    import pdb; pdb.set_trace()
    current = list(robot.get_current_pose())
    above_pregrasp = pose_pregrasp.copy()
    above_pregrasp[2] = max(args.safe_height, pose_pregrasp[2] + 0.1)  # 保持在安全高度或更高
    
    if not move_smooth(robot, above_pregrasp, current, 
                       velocity=args.transit_velocity, 
                       num_waypoints=args.num_waypoints):
        print("Failed to reach above pre-grasp, aborting...")
        robot.disconnect()
        return
    
    import pdb; pdb.set_trace()
    print("\n[Step 3] Descending to pre-grasp position...")
    current = list(robot.get_current_pose())
    if not move_smooth(robot, pose_pregrasp, current,
                       velocity=args.approach_velocity,
                       num_waypoints=2): 
        print("Failed to reach pre-grasp position, aborting...")
        robot.disconnect()
        return
    
    time.sleep(0.3)  # 稳定
    
    import pdb; pdb.set_trace()
    print("\n[Step 5] Final approach to grasp position...")
    current = list(robot.get_current_pose())
    if not move_smooth(robot, pose_grasp, current,
                       velocity=args.approach_velocity / 2,  # 更慢的速度
                       num_waypoints=1):
        print("Warning: Failed to reach exact grasp position, grasping at current position...")

    hand.finger_move([103, 0, 160, 143, 135, 131, 255, 255, 255, 255])
    # r = 15圆柱
    # hand.finger_move([103, 53, 160, 143, 135, 131, 255, 255, 255, 255])
    
    robot.disconnect()


if __name__ == "__main__":
    main()


