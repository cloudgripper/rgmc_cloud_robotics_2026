import cv2
import numpy as np
import os
import glob
import yaml
import time

def undistort_image_path(K, D, img_path):
    """
        Function to undistort an image using the given camera matrix and distortion coefficients.
        
        Args:
            K, D: Camera matrix and distortion coefficients.
            img_path (str): The path to the image to undistort.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"Failed to load image at {img_path}")

    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img

def undistort_image(K, D, img):
    """
        Function to undistort an image using the given camera matrix and distortion coefficients.
    """
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img
    
def load_parameters_from_file(filename):
    """
    Load the camera parameters from a YAML file.

    Parameters:
    filename (str): The file name where the parameters are saved.

    Returns:
    tuple: The camera matrix and distortion coefficients.
    """
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
        K = np.array(data['K'])
        D = np.array(data['D'])
    return K, D

if __name__ == '__main__':
    pass    