import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# 创建保存目录
dataset_dir = "custom_dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 配置RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始流
pipeline.start(config)

try:
    frame_count = 0
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # 转换为numpy数组
        image = np.asanyarray(color_frame.get_data())

        # depth_frame = frames.get_depth_frame()
        # if not depth_frame:
        #     continue
        
        # image = np.asanyarray(depth_frame.get_data())
        # image = cv2.convertScaleAbs(image, alpha=0.03)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 显示图像
        cv2.imshow('Data Collection', image)
        
        key = cv2.waitKey(1)
        if key == ord('s'):  # 按's'保存图像
            img_name = f"image_{frame_count:04d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            cv2.imwrite(img_path, image)
            print(f"Saved {img_path}")
            frame_count += 1
        elif key == ord('q'):  # 按'q'退出
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()