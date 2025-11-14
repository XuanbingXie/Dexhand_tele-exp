from estimater import *
import argparse
import pyrealsense2 as rs
import cv2
import json
import time


def get_mask_from_user(color, depth):
    roi = cv2.selectROI("Select Object Region (press ENTER)", color[...,::-1], False, False)
    cv2.destroyWindow("Select Object Region (press ENTER)")
    mask = np.zeros((color.shape[0], color.shape[1]), dtype=bool)
    x, y, w, h = [int(v) for v in roi]
    if w > 0 and h > 0:
        mask[y:y+h, x:x+w] = True
        depth_roi = depth[y:y+h, x:x+w]
        valid_depth = depth_roi[depth_roi > 0]
        if len(valid_depth) > 0:
            depth_mean = np.median(valid_depth)
            depth_threshold = 0.15
            mask = mask & (np.abs(depth - depth_mean) < depth_threshold) & (depth > 0)
    return mask


def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    print("\nAvailable RealSense devices:")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.get_info(rs.camera_info.name)} - Serial: {dev.get_info(rs.camera_info.serial_number)}")
    return devices


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--left_serial', type=str, default=None)
  parser.add_argument('--right_serial', type=str, default=None)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--pose_output', type=str, default=f'{code_dir}/pose_output.txt')
  args = parser.parse_args()

  devices = list_realsense_devices()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  debug = args.debug
  debug_dir = args.debug_dir
  os.makedirs(debug_dir, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  use_dual_camera = args.left_serial is not None and args.right_serial is not None

  if use_dual_camera:
    pipeline_left = rs.pipeline()
    config_left = rs.config()
    config_left.enable_device(args.left_serial)
    config_left.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_left.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile_left = pipeline_left.start(config_left)
    align_left = rs.align(rs.stream.color)
    depth_scale_left = profile_left.get_device().first_depth_sensor().get_depth_scale()
    intr_left = profile_left.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K_left = np.array([[intr_left.fx, 0, intr_left.ppx], [0, intr_left.fy, intr_left.ppy], [0, 0, 1]])

  pipeline_right = rs.pipeline()
  config_right = rs.config()
  if args.right_serial:
    config_right.enable_device(args.right_serial)
  config_right.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config_right.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  profile_right = pipeline_right.start(config_right)
  align_right = rs.align(rs.stream.color)
  depth_scale_right = profile_right.get_device().first_depth_sensor().get_depth_scale()
  intr_right = profile_right.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  K_right = np.array([[intr_right.fx, 0, intr_right.ppx], [0, intr_right.fy, intr_right.ppy], [0, 0, 1]])

  pose = None
  frame_count = 0
  init_done = False

  try:
    while True:
      if use_dual_camera and not init_done:
        frames_left = pipeline_left.wait_for_frames()
        frames_left = align_left.process(frames_left)
        color_frame = frames_left.get_color_frame()
        depth_frame = frames_left.get_depth_frame()
        if not color_frame or not depth_frame:
          continue
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) * depth_scale_left
        K = K_left
        cam_name = "Left"
      else:
        frames_right = pipeline_right.wait_for_frames()
        frames_right = align_right.process(frames_right)
        color_frame = frames_right.get_color_frame()
        depth_frame = frames_right.get_depth_frame()
        if not color_frame or not depth_frame:
          continue
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) * depth_scale_right
        K = K_right
        cam_name = "Right"

      if not init_done:
        mask = get_mask_from_user(color, depth)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        init_done = True
        logging.info(f"Initial pose registered from {cam_name} camera")
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

      if pose is not None:
        with open(args.pose_output, 'w') as f:
          json.dump({'pose': pose.tolist(), 'timestamp': time.time()}, f)
        
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow(f'{cam_name} Camera', vis[...,::-1])
        
      key = cv2.waitKey(1)
      if key == 27 or key == ord('q'):
        break
      elif key == ord('r'):
        init_done = False
        logging.info("Reset - will re-register on next frame")

      frame_count += 1

  finally:
    pipeline_right.stop()
    if use_dual_camera:
      pipeline_left.stop()
    cv2.destroyAllWindows()
