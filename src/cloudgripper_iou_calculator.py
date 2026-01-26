import cv2
import numpy as np
import calib_fisheye_cam as calib_fisheye_cam
from shapes import SQUARE_CORNERS, CIRCLE_CORNERS, T_CORNERS

class ICRA_2026_IOU_Calculator:
    def __init__(self, param_file):
        self.camera_matrix, self.distortion_coeffs = calib_fisheye_cam.load_parameters_from_file(param_file)

    def undistort_image(self, image_bgr):
        return calib_fisheye_cam.undistort_image(self.camera_matrix, self.distortion_coeffs, image_bgr)

    def create_hsv_mask(self, image_bgr, color_low_hsv, color_high_hsv):
        image_hsv = cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2HSV)
        return cv2.inRange(image_hsv, color_low_hsv, color_high_hsv)

    def find_largest_contour(self, image_bgr):
        # find contours
        contours, _ = cv2.findContours(image_bgr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=lambda c: cv2.contourArea(c))
        return largest_contour, contours
    
    def draw_contours_on_image(self, image_bgr, contours, color_bgr=(0, 0, 255), fill=False):
        contour_image = cv2.drawContours(image_bgr.copy(), contours, -1, color_bgr, 2)
        if fill:
            contour_image = cv2.fillPoly(contour_image.copy(), pts=contours, color=color_bgr)
        return contour_image

    def shift_contour(self, object_contour, shift_x=0, shift_y=0):
        return object_contour + np.array([shift_x, shift_y])
    
    def compute_contour_intersection(self, object_contour, target_contour, image_shape):
        # Create a binary mask for each contour
        mask_object = np.zeros(image_shape, dtype=np.uint8)
        mask_target = np.zeros(image_shape, dtype=np.uint8)

        # Fill contours on masks
        cv2.drawContours(mask_object, [object_contour], -1, 255, -1)
        cv2.drawContours(mask_target, [target_contour], -1, 255, -1)

        # Bitwise AND to get the intersection
        intersection_mask = cv2.bitwise_and(mask_object, mask_target)
        
        # Check if these is any intersection
        if cv2.countNonZero(intersection_mask) == 0:
            return False, None

        # Find all contours of the intersection
        _, intersection_contours = self.find_largest_contour(intersection_mask)

        if len(intersection_contours) > 0:
            return True, intersection_contours
        
        return False, None

    def calculate_iou(self, object_contour, target_contour, intersection_contours):
        # Find the area of the object contour
        object_area = cv2.contourArea(object_contour)
        # Find the area of the target contour
        target_area = cv2.contourArea(target_contour)
        # Find the area of the intersection contours
        if intersection_contours is not None:
            intersection_area = sum(cv2.contourArea(contour) for contour in intersection_contours)
        else:
            intersection_area = 0.0
        # Find the IOU
        union_area = object_area + target_area - intersection_area
        iou_score = intersection_area / union_area
        return iou_score

    def calculate_iou_from_camera_image(self, image_base_undistorted, object_corners):
        image_shape = (image_base_undistorted.shape[0], image_base_undistorted.shape[1])

        # Find the red mask in the base image
        red_mask = self.create_hsv_mask(image_base_undistorted, (0, 100, 0), (7, 255, 255))  
        # Find the largest contour in the red mask
        object_contour, _ = self.find_largest_contour(red_mask)
        # Create target contour
        target_contour = self.project_object_to_image(object_corners)

        estimated_object_contour = self.align_template_contour_to_detected(
            object_contour,
            target_contour,
            image_shape,
            debug=False,
        )
        object_contour = estimated_object_contour

        # Create intersection contour
        has_intersection, intersection_contours = self.compute_contour_intersection(object_contour, target_contour, image_shape)

        # create image of input with contours overlayed
        target_contour_image = self.draw_contours_on_image(image_base_undistorted.copy(), [target_contour], color_bgr=(0, 255, 0))
        intersection_contour_image = self.draw_contours_on_image(target_contour_image.copy(), intersection_contours, color_bgr=(255, 255, 255), fill=True)
        
        # Find the IOU
        iou_score = self.calculate_iou(object_contour, target_contour, intersection_contours)
        return iou_score, intersection_contour_image

    def project_object_to_image(self, object_points, offset_from_center_mm=[-40, 30], rotation_angle_rad=0):
        # Distance to glass plate (Z axis) in mm
        distance_to_glass = 157.64

        # Rotation Vector (rotation_vector): 
        # If camera is looking straight up at a plexiglass plate
        rotation_vector = np.zeros((3, 1), dtype=np.float32)
        rotation_vector[2] = rotation_angle_rad

        # Translation Vector (translation_vector):
        x_offset_mm = offset_from_center_mm[0]
        y_offset_mm = offset_from_center_mm[1]
        translation_vector = np.array([[x_offset_mm], [y_offset_mm], [distance_to_glass]], dtype=np.float32)

        # Calculate projection
        image_points, jacobian = cv2.projectPoints(object_points, rotation_vector, translation_vector, self.camera_matrix, np.zeros((4, 1)))

        contour_points = image_points.astype(np.int32)
        return contour_points

    def align_template_contour_to_detected(self, object_contour, target_contour, image_shape_hw, *, number_of_iterations=200, termination_eps=1e-6, debug=False):
        """
        Align target_contour to object_contour using cv2.findTransformECC.
        """
        H, W = int(image_shape_hw[0]), int(image_shape_hw[1])

        # Create binary masks
        template_mask = np.zeros((H, W), dtype=np.uint8)
        detected_mask = np.zeros((H, W), dtype=np.uint8)

        cv2.drawContours(template_mask, [target_contour], -1, 255, thickness=-1)
        cv2.drawContours(detected_mask, [object_contour], -1, 255, thickness=-1)

        template_f = (template_mask.astype(np.float32) / 255.0)
        detected_f = (detected_mask.astype(np.float32) / 255.0)

        # Compute centroids
        M_target = cv2.moments(target_contour)
        M_detected = cv2.moments(object_contour)
        
        if M_target["m00"] > 0 and M_detected["m00"] > 0:
            cx_target = M_target["m10"] / M_target["m00"]
            cy_target = M_target["m01"] / M_target["m00"]
            cx_detected = M_detected["m10"] / M_detected["m00"]
            cy_detected = M_detected["m01"] / M_detected["m00"]
            
            # Initialize warp matrix with translation to align centroids (tx, ty)
            tx = cx_detected - cx_target
            ty = cy_detected - cy_target
        else:
            tx, ty = 0.0, 0.0

        # warp matrix is 2x3
        warp_matrix = np.array([[1.0, 0.0, tx],
                                [0.0, 1.0, ty]], dtype=np.float32)

        # termination criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(number_of_iterations),
            float(termination_eps),
        )

        # run ECC alignment
        try:
            cc, warp_matrix = cv2.findTransformECC(
                template_f,
                detected_f,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN, # rotation + translation (no scale)
                criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error as e:
            print("e = ",e)
            return object_contour.copy().astype(np.int32)


        target_pts = np.asarray(target_contour, dtype=np.float32)
        estimated_pts = cv2.transform(target_pts, warp_matrix)

        estimated_contour = estimated_pts.astype(np.int32)

        # debug visualization
        if debug:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.drawContours(vis, [object_contour], -1, (0, 255, 0), 2)     # detected = green
            cv2.drawContours(vis, [estimated_contour], -1, (255, 0, 0), 2)    # aligned template = blue
            cv2.imshow("ECC alignment (green=detected, blue=aligned template)", vis)
            cv2.waitKey(0)

        return estimated_contour

if __name__ == "__main__":
    image_base_bgr = cv2.imread("sample_images/base_image_cr01_square_2.png")

    IOU_evaluator = ICRA_2026_IOU_Calculator("config/base-camera-calibration/camera-params-cr01.yaml")
    image_base_undistorted = IOU_evaluator.undistort_image(image_base_bgr)
    frame_score, vis_image = IOU_evaluator.calculate_iou_from_camera_image(image_base_undistorted, SQUARE_CORNERS)

    print(f"frame_score = ", frame_score)
    cv2.imshow("vis", vis_image)
    cv2.waitKey(0)