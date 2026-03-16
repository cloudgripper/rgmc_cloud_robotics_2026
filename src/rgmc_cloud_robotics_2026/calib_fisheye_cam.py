import cv2
import numpy as np
import yaml


def undistort_image_path(K, D, img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"Failed to load image at {img_path}")

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def undistort_image(K, D, img):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def load_parameters_from_file(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
        K = np.array(data['K'])
        D = np.array(data['D'])
    return K, D


def distort_point(K, D, point):
    x, y = point
    x_norm = (x - K[0, 2]) / K[0, 0]
    y_norm = (y - K[1, 2]) / K[1, 1]
    obj_point = np.array([[[x_norm, y_norm, 1.0]]], dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    distorted, _ = cv2.fisheye.projectPoints(obj_point, rvec, tvec, K, D)
    return [float(distorted[0][0][0]), float(distorted[0][0][1])]


def undistort_contour(K, D, contour):
    original_shape = contour.shape
    points = contour.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.fisheye.undistortPoints(points, K, D, P=K)
    return undistorted.reshape(-1, 2).astype(np.int32).reshape(original_shape)


def distort_contour(K, D, contour):
    original_shape = contour.shape
    points = contour.reshape(-1, 2)
    distorted = np.array([distort_point(K, D, p) for p in points], dtype=np.int32)
    return distorted.reshape(original_shape)
