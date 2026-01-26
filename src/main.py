import cv2
import numpy as np
from shapes import SQUARE_CORNERS, CIRCLE_CORNERS, T_CORNERS
from cloudgripper_iou_calculator import ICRA_2026_IOU_Calculator

image_base_bgr = cv2.imread("sample_images/base_image_cr01_square_2.png")

IOU_evaluator = ICRA_2026_IOU_Calculator("config/base-camera-calibration/camera-params-cr01.yaml")
image_base_undistorted = IOU_evaluator.undistort_image(image_base_bgr)
frame_score, vis_image = IOU_evaluator.calculate_iou_from_camera_image(image_base_undistorted, SQUARE_CORNERS)

print(f"frame_score = ", frame_score)
cv2.imshow("vis", vis_image)
cv2.waitKey(0)