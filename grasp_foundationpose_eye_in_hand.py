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


def detect_pose_with_foundationpose(mesh_file: str, debug_dir: str = "fp_debug") -> Tuple[np.ndarray, dict]:
    """
    Returns:
      pose_cam_obj (4x4): object pose in camera frame (T_cam_obj)
      calib: dict with K and other info
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
                # Optional visualization
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow("FoundationPose", vis[..., ::-1])
                cv2.waitKey(500)
                cv2.destroyAllWindows()
                return pose, {
                    "K": K,
                    "depth_scale": depth_scale,
                    "mesh_to_center": np.linalg.inv(to_origin),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--hand_eye", type=str, required=True, help="JSON file containing {'T_ee_cam': [[...],[...],[...],[...]]}")
    parser.add_argument("--pregrasp_height", type=float, default=0.01, help="pre-grasp height above object in RM Z direction (m)")
    parser.add_argument("--grasp_offset", type=float, default=0.0, help="final Z offset from object position for grasp (m)")
    parser.add_argument("--hand_speed", type=int, default=120)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    robot = RobotArmController("192.168.1.28", 8080, 3)
    hand = LinkerHandApi(hand_type="right", hand_joint="L10")
    hand.set_speed(speed=[120,200,200,200,200])

    point = [0.55, -0.12, -0.03, 3.0, -1.1, 0.9]
    robot.movel(point, v = 40)
    hand.finger_move([255, 255, 255, 255, 255, 255, 255, 255, 255, 255])

    pose_cam_obj, info = detect_pose_with_foundationpose(args.mesh_file, debug_dir=os.path.join(ROOT_DIR, "fp_debug"))
    mesh_to_center = info.get("mesh_to_center", np.eye(4, dtype=np.float64))
    T_cam_obj = pose_cam_obj @ mesh_to_center

    current_pose = robot.get_current_pose()
    x, y, z, rx, ry, rz = current_pose

    # Build T_base_ee from current pose (assume rx,ry,rz is ZYX euler in radians)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    R_base_ee = Rz @ Ry @ Rx
    T_base_ee = np.eye(4, dtype=np.float64)
    T_base_ee[:3, :3] = R_base_ee
    T_base_ee[:3, 3] = [x, y, z]

    T_ee_cam = load_hand_eye_transform(args.hand_eye)
    
    # Step 1: Convert camera coordinate system to RM coordinate system
    T_rm_cam = get_camera_to_rm_transform()
    
    # Step 2: Convert T_cam_obj from OpenCV camera frame to RM camera frame
    T_cam_obj_rm = T_rm_cam @ T_cam_obj
    
    # Step 3: Convert T_ee_cam to RM frame
    T_ee_cam_rm = T_rm_cam @ T_ee_cam
    
    # Step 4: Compute object pose in base frame (绝对位置)
    T_cam_ee_rm = np.linalg.inv(T_ee_cam_rm)
    T_ee_obj = T_cam_ee_rm @ T_cam_obj_rm
    T_base_obj = T_base_ee @ T_ee_obj

    # 5) Define grasp strategy: move to pre-grasp above object, then to grasp position
    R_base_tool = R_base_ee
    p_obj = T_base_obj[:3, 3]

    # RM坐标系：X前，Y左，Z上，所以"上方"是+Z方向
    p_pregrasp = p_obj + np.array([0, 0, args.pregrasp_height])  # 沿Z轴向上
    
    # Grasp pose: at object position (or with small offset if needed)
    p_grasp = p_obj + np.array([0, 0, args.grasp_offset])  # 可以沿Z轴微调

    rx_cmd, ry_cmd, rz_cmd = rmat_to_rvec_zyx(R_base_tool)
    pose_pregrasp = [float(p_pregrasp[0]), float(p_pregrasp[1])-0.05, -0.15, rx_cmd, ry_cmd, rz_cmd]
    pose_grasp = [float(p_grasp[0]), float(p_grasp[1]), float(p_grasp[2]), rx_cmd, ry_cmd, rz_cmd]

    # 6) Execute motion and grasp
    # Open before approach (for parallel jaw hands; for LinkerHand, use a spread pose)
    
    print(pose_pregrasp)
    import pdb; pdb.set_trace()
    robot.movel(pose_pregrasp, v=25)
    import pdb; pdb.set_trace()
    hand.finger_move([103, 53, 160, 143, 135, 131, 255, 255, 255, 255])
    # robot.movel(pose_pregrasp, v=10)

    robot.disconnect()


if __name__ == "__main__":
    main()


