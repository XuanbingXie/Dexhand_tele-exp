from estimater import *
import argparse
import pyrealsense2 as rs
import cv2


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
            depth_threshold = 0.1
            mask = mask & (np.abs(depth - depth_mean) < depth_threshold) & (depth > 0)
    return mask


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/xxbdata/mesh/r=20.obj')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

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

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  profile = pipeline.start(config)

  align = rs.align(rs.stream.color)
  depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

  intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  K = np.array([[intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]])

  pose = None
  frame_count = 0

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

      if frame_count == 0:
        mask = get_mask_from_user(color, depth)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        logging.info("Initial pose registered")
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

      if pose is not None:
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('FoundationPose RealSense', vis[...,::-1])
        
      key = cv2.waitKey(1)
      if key == 27 or key == ord('q'):
        break
      elif key == ord('r'):
        frame_count = -1
        logging.info("Reset - will re-register on next frame")

      frame_count += 1

  finally:
    pipeline.stop()
    cv2.destroyAllWindows()
