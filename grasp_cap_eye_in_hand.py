import os
import sys
import json
import time
import argparse
from typing import Tuple
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import pyrealsense2 as rs  # type: ignore
import cv2  # type: ignore

from RM_API2.RealMan.realman import RobotArmController  # type: ignore
from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore


def load_hand_eye_transform(hand_eye_path: str) -> np.ndarray:
    with open(hand_eye_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    T = np.asarray(data["T_ee_cam"], dtype=np.float64)
    assert T.shape == (4, 4), "T_ee_cam must be 4x4"
    return T


def rmat_to_rvec_zyx(R: np.ndarray) -> Tuple[float, float, float]:
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
    if w > 0 and h > 0:
        mask[y : y + h, x : x + w] = True
        depth_roi = depth[y : y + h, x : x + w]
        valid_depth = depth_roi[depth_roi > 0]
        if len(valid_depth) > 0:
            depth_mean = np.median(valid_depth)
            depth_threshold = 0.02
            mask = mask & (np.abs(depth - depth_mean) < depth_threshold) & (depth > 0)
    return mask


def mask_to_point_cloud(mask: np.ndarray, depth: np.ndarray, intr: rs.intrinsics) -> np.ndarray:
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth[ys, xs]
    x = (xs - intr.ppx) * z / intr.fx
    y = (ys - intr.ppy) * z / intr.fy
    pts = np.stack([x, y, z], axis=1)
    return pts.astype(np.float64)


def fit_plane_normal(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([0, 0, 1], dtype=np.float64)
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-9)


def detect_cap_pose_from_depth(debug_dir: str = "cap_debug") -> Tuple[np.ndarray, dict]:
    os.makedirs(debug_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
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
            cloud = mask_to_point_cloud(mask, depth, intr)
            if len(cloud) < 50:
                continue

            centroid = cloud.mean(axis=0)
            normal = fit_plane_normal(cloud)

            x_guess = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x_guess, normal)) > 0.9:
                x_guess = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(normal, x_guess)
            if np.linalg.norm(y_axis) < 1e-6:
                y_axis = np.array([0.0, 1.0, 0.0])
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, normal)
            x_axis /= np.linalg.norm(x_axis)

            R = np.eye(3, dtype=np.float64)
            R[:, 0] = x_axis
            R[:, 1] = y_axis
            R[:, 2] = normal

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R
            pose[:3, 3] = centroid
            return pose, {"intr": intr, "depth_scale": depth_scale}
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def get_camera_to_rm_transform() -> np.ndarray:
    R_rm_cam = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    T_rm_cam = np.eye(4, dtype=np.float64)
    T_rm_cam[:3, :3] = R_rm_cam
    return T_rm_cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_eye", type=str, required=True)
    parser.add_argument("--pregrasp_height", type=float, default=0.02)
    parser.add_argument("--grasp_offset", type=float, default=0.0)
    parser.add_argument("--hand_speed", type=int, default=120)
    args = parser.parse_args()

    robot = RobotArmController("192.168.1.28", 8080, 3)
    hand = LinkerHandApi(hand_type="right", hand_joint="L10")
    hand.set_speed(speed=[args.hand_speed, 200, 200, 200, 200])

    home_pose = [0.55, -0.12, -0.03, 3.0, -1.1, 0.9]
    robot.movel(home_pose, v=40)
    hand.finger_move([255] * 10)

    pose_cam_obj, _ = detect_cap_pose_from_depth(debug_dir=os.path.join(ROOT_DIR, "cap_debug"))
    T_cam_obj = pose_cam_obj

    current_pose = robot.get_current_pose()
    x, y, z, rx, ry, rz = current_pose

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    R_base_ee = Rz @ Ry @ Rx_mat
    T_base_ee = np.eye(4, dtype=np.float64)
    T_base_ee[:3, :3] = R_base_ee
    T_base_ee[:3, 3] = [x, y, z]

    T_ee_cam = load_hand_eye_transform(args.hand_eye)

    T_rm_cam = get_camera_to_rm_transform()
    T_cam_obj_rm = T_rm_cam @ T_cam_obj
    T_ee_cam_rm = T_rm_cam @ T_ee_cam

    T_cam_ee_rm = np.linalg.inv(T_ee_cam_rm)
    T_ee_obj = T_cam_ee_rm @ T_cam_obj_rm
    T_base_obj = T_base_ee @ T_ee_obj

    p_obj = T_base_obj[:3, 3]
    p_pregrasp = p_obj + np.array([0, 0, args.pregrasp_height])
    p_grasp = p_obj + np.array([0, 0, args.grasp_offset])

    rx_cmd, ry_cmd, rz_cmd = rmat_to_rvec_zyx(R_base_ee)
    pose_pregrasp = [float(p_pregrasp[0]), float(p_pregrasp[1]), float(p_pregrasp[2]), rx_cmd, ry_cmd, rz_cmd]
    pose_grasp = [float(p_grasp[0]), float(p_grasp[1]), float(p_grasp[2]), rx_cmd, ry_cmd, rz_cmd]

    robot.movel(pose_pregrasp, v=25)
    robot.movel(pose_grasp, v=10)
    hand.finger_move([103, 53, 160, 143, 135, 131, 255, 255, 255, 255])
    robot.movel(pose_pregrasp, v=25)

    robot.disconnect()


if __name__ == "__main__":
    main()

