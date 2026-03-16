import cv2
import numpy as np
from . import calib_fisheye_cam
from .shapes import SQUARE_CORNERS, CIRCLE_CORNERS, T_CORNERS


class PushingTaskIOU_Calculator:

    _RED_HSV_LOW   = (170, 100,   0)
    _RED_HSV_HIGH  = (  7, 255, 255)
    _RED_HSV_LOW2  = (170, 100,  50)
    _RED_HSV_HIGH2 = (179, 255, 150)

    def __init__(self, param_file, distance_to_glass_mm=157.64):
        self.camera_matrix, self.distortion_coeffs = calib_fisheye_cam.load_parameters_from_file(param_file)
        self.distance_to_glass_mm = distance_to_glass_mm

    def undistort_image(self, img_bgr):
        return calib_fisheye_cam.undistort_image(self.camera_matrix, self.distortion_coeffs, img_bgr)

    def distort_point(self, point):
        return calib_fisheye_cam.distort_point(self.camera_matrix, self.distortion_coeffs, point)

    def undistort_contour(self, contour):
        return calib_fisheye_cam.undistort_contour(self.camera_matrix, self.distortion_coeffs, contour)

    def distort_contour(self, contour):
        return calib_fisheye_cam.distort_contour(self.camera_matrix, self.distortion_coeffs, contour)

    def create_hsv_mask(self, img_bgr, color_low_hsv, color_high_hsv,
                        color_low_hsv2=None, color_high_hsv2=None):
        img_hsv = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2HSV)
        low  = np.array(color_low_hsv,  dtype=np.uint8)
        high = np.array(color_high_hsv, dtype=np.uint8)

        if low[0] <= high[0]:
            return cv2.inRange(img_hsv, low, high)

        if color_low_hsv2 is not None and color_high_hsv2 is not None:
            upper_low  = np.array(color_low_hsv2,  dtype=np.uint8)
            upper_high = np.array(color_high_hsv2, dtype=np.uint8)
        else:
            upper_low  = np.array([low[0], low[1],  low[2]],  dtype=np.uint8)
            upper_high = np.array([179,    high[1], high[2]], dtype=np.uint8)

        lower_low  = np.array([0,       low[1],  low[2]],  dtype=np.uint8)
        lower_high = np.array([high[0], high[1], high[2]], dtype=np.uint8)

        return cv2.bitwise_or(
            cv2.inRange(img_hsv, upper_low, upper_high),
            cv2.inRange(img_hsv, lower_low, lower_high),
        )

    def k_means_darker_mask(self, img_bgr, mask, cluster_distance_threshold=0.2):
        if mask is None or np.sum(mask) == 0:
            return mask

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_pixels = img_hsv[mask > 0]
        if len(mask_pixels) == 0:
            return mask

        v_pixels = mask_pixels[:, 2].astype(np.float32).reshape(-1, 1) / 255.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(v_pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        if np.linalg.norm(centers[0] - centers[1]) < cluster_distance_threshold:
            return mask

        labels_flat = labels.flatten()
        mask_class1 = np.zeros_like(mask)
        mask_class2 = np.zeros_like(mask)
        y_coords, x_coords = np.where(mask > 0)
        for idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            if labels_flat[idx] == 0:
                mask_class1[y, x] = 255
            else:
                mask_class2[y, x] = 255

        return mask_class1 if centers[0][0] < centers[1][0] else mask_class2

    def find_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour, contours

    def draw_contours_on_image(self, img_bgr, contours, color_bgr=(0, 0, 255), fill=False):
        result = cv2.drawContours(img_bgr.copy(), contours, -1, color_bgr, 2)
        if fill:
            result = cv2.fillPoly(result, pts=contours, color=color_bgr)
        return result

    def shift_contour(self, contour, shift_x=0, shift_y=0):
        return contour + np.array([shift_x, shift_y])

    def compute_contour_intersection(self, obj_contour, target_contour, img_shape):
        mask_obj    = np.zeros(img_shape, dtype=np.uint8)
        mask_target = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(mask_obj,    [obj_contour],    -1, 255, -1)
        cv2.drawContours(mask_target, [target_contour], -1, 255, -1)
        intersection_mask = cv2.bitwise_and(mask_obj, mask_target)
        if cv2.countNonZero(intersection_mask) == 0:
            return False, None
        _, intersection_contours = self.find_largest_contour(intersection_mask)
        if len(intersection_contours) > 0:
            return True, intersection_contours
        return False, None

    def calculate_iou(self, obj_contour, target_contour, intersection_contours):
        obj_area    = cv2.contourArea(obj_contour)
        target_area = cv2.contourArea(target_contour)
        intersection_area = (
            sum(cv2.contourArea(c) for c in intersection_contours)
            if intersection_contours is not None else 0.0
        )
        union_area = obj_area + target_area - intersection_area
        return intersection_area / union_area

    def project_object_to_image(self, obj_points, offset_from_center_mm=[-40, 30], rotation_angle_rad=0):
        rotation_vector = np.zeros((3, 1), dtype=np.float32)
        rotation_vector[2] = rotation_angle_rad
        translation_vector = np.array(
            [[offset_from_center_mm[0]], [offset_from_center_mm[1]], [self.distance_to_glass_mm]],
            dtype=np.float32,
        )
        img_points, _ = cv2.projectPoints(
            obj_points, rotation_vector, translation_vector,
            self.camera_matrix, np.zeros((4, 1)),
        )
        return img_points.astype(np.int32)

    def align_template_contour_to_detected(self, obj_contour, target_contour, img_shape_hw, *,
                                            coarse_step_deg=5, number_of_iterations=200,
                                            termination_eps=1e-6, max_center_dist_px=20,
                                            debug=False, debug_img=None):
        H, W = int(img_shape_hw[0]), int(img_shape_hw[1])

        M_target   = cv2.moments(target_contour)
        M_detected = cv2.moments(obj_contour)
        if M_target["m00"] > 0 and M_detected["m00"] > 0:
            cx_target   = M_target["m10"]  / M_target["m00"]
            cy_target   = M_target["m01"]  / M_target["m00"]
            cx_detected = M_detected["m10"] / M_detected["m00"]
            cy_detected = M_detected["m01"] / M_detected["m00"]
        else:
            cx_target = cy_target = cx_detected = cy_detected = 0.0

        if debug:
            dbg_img = debug_img.copy() if debug_img is not None else np.zeros((H, W, 3), dtype=np.uint8)
            cv2.drawContours(dbg_img, [target_contour], -1, (100, 100, 100), 1)
            cv2.drawContours(dbg_img, [obj_contour],    -1, (100, 100, 100), 1)
            cv2.circle(dbg_img, (int(cx_target),   int(cy_target)),   6, (0, 255, 0), -1)
            cv2.circle(dbg_img, (int(cx_detected), int(cy_detected)), 6, (0, 0, 255), -1)
            cv2.putText(dbg_img, "target centroid",
                        (int(cx_target)+8, int(cy_target)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(dbg_img, "detected centroid",
                        (int(cx_detected)+8, int(cy_detected)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imshow("centroids", dbg_img)
            cv2.waitKey(0)

        coarse_results = []
        for angle_deg in np.arange(0, 360, coarse_step_deg):
            warp      = self._build_warp(cx_target, cy_target, cx_detected, cy_detected, np.deg2rad(angle_deg))
            candidate = self._apply_warp_to_contour(target_contour, warp)
            iou       = self._contour_iou(obj_contour, candidate, H, W)
            coarse_results.append((iou, angle_deg, candidate, warp))

        coarse_results.sort(key=lambda x: x[0], reverse=True)
        _, _, best_coarse_contour, best_coarse_warp = coarse_results[0]

        detected_mask        = np.zeros((H, W), dtype=np.uint8)
        template_mask_coarse = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(detected_mask,        [obj_contour],         -1, 255, thickness=-1)
        cv2.drawContours(template_mask_coarse, [best_coarse_contour], -1, 255, thickness=-1)
        detected_f        = detected_mask.astype(np.float32)        / 255.0
        template_coarse_f = template_mask_coarse.astype(np.float32) / 255.0

        warp_delta = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(number_of_iterations), float(termination_eps))
        try:
            _, warp_delta = cv2.findTransformECC(
                template_coarse_f, detected_f, warp_delta,
                cv2.MOTION_EUCLIDEAN, criteria, inputMask=None, gaussFiltSize=5,
            )
            estimated_contour = self._apply_warp_to_contour(best_coarse_contour, warp_delta)
            final_warp        = self._compose_euclidean_warps(warp_delta, best_coarse_warp)
        except cv2.error:
            estimated_contour = best_coarse_contour
            final_warp        = best_coarse_warp

        def _centroid(c):
            M = cv2.moments(c)
            return (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] > 0 else (0.0, 0.0)

        est_cx, est_cy = _centroid(estimated_contour)
        if np.hypot(est_cx - cx_detected, est_cy - cy_detected) > max_center_dist_px:
            retry_step    = max(1.0, coarse_step_deg / 2.0)
            retry_results = []
            for angle_deg in np.arange(0, 360, retry_step):
                warp      = self._build_warp(cx_target, cy_target, cx_detected, cy_detected, np.deg2rad(angle_deg))
                candidate = self._apply_warp_to_contour(target_contour, warp)
                iou       = self._contour_iou(obj_contour, candidate, H, W)
                retry_results.append((iou, angle_deg, candidate, warp))
            retry_results.sort(key=lambda x: x[0], reverse=True)
            _, _, estimated_contour, final_warp = retry_results[0]

        return estimated_contour, final_warp

    def calculate_iou_from_undistorted_base(self, img_base_undistorted, target_contour,
                                             obj_corners, debug=False):
        undistorted_shape = (img_base_undistorted.shape[0], img_base_undistorted.shape[1])

        template_contour = self.project_object_to_image(obj_corners, [0, 0], 0)

        obj_contour = self._detect_red_object_contour(img_base_undistorted, apply_kmeans=True)

        obj_contour, alignment_warp = self.align_template_contour_to_detected(
            obj_contour, template_contour, undistorted_shape, debug=debug,
        )
        alignment_angle_deg = float(np.degrees(np.arctan2(alignment_warp[1, 0], alignment_warp[0, 0])))

        _, intersection_contours = self.compute_contour_intersection(
            obj_contour, target_contour, undistorted_shape,
        )
        iou_score = self.calculate_iou(obj_contour, target_contour, intersection_contours)

        vis_img = self.draw_contours_on_image(img_base_undistorted.copy(), [target_contour], color_bgr=(0, 255, 0))
        vis_img = self.draw_contours_on_image(vis_img, intersection_contours, color_bgr=(255, 255, 255), fill=True)

        return iou_score, vis_img, obj_contour, target_contour, alignment_angle_deg

    def generate_target_contour(self, obj_corners, offset_from_center_mm=[0, 0], rotation_angle_rad=0):
        return self.project_object_to_image(obj_corners, offset_from_center_mm, rotation_angle_rad)

    def calculate_iou_from_distorted_base(self, img_base_distorted,
                                           target_contour_undistorted,
                                           obj_corners,
                                           debug=False):
        img_base_undistorted = self.undistort_image(img_base_distorted)
        undistorted_shape    = (img_base_undistorted.shape[0], img_base_undistorted.shape[1])

        template_contour_undistorted = self.project_object_to_image(obj_corners, [0, 0], 0)

        obj_contour_distorted   = self._detect_red_object_contour(img_base_distorted, apply_kmeans=True)
        obj_contour_undistorted = self.undistort_contour(obj_contour_distorted)

        obj_contour_undistorted, alignment_warp = self.align_template_contour_to_detected(
            obj_contour_undistorted, template_contour_undistorted, undistorted_shape,
            debug=debug, debug_img=img_base_undistorted.copy(),
        )
        alignment_angle_deg = float(np.degrees(np.arctan2(alignment_warp[1, 0], alignment_warp[0, 0])))
        if alignment_angle_deg < 0:
            alignment_angle_deg += 180

        _, intersection_contours_undistorted = self.compute_contour_intersection(
            obj_contour_undistorted, target_contour_undistorted, undistorted_shape,
        )
        iou_score = self.calculate_iou(
            obj_contour_undistorted, target_contour_undistorted, intersection_contours_undistorted,
        )

        target_contour_distorted = self.distort_contour(target_contour_undistorted)
        obj_contour_distorted    = self.distort_contour(obj_contour_undistorted)
        intersection_contours_distorted = (
            [self.distort_contour(c) for c in intersection_contours_undistorted]
            if intersection_contours_undistorted is not None else None
        )

        vis_img_distorted = self.draw_contours_on_image(
            img_base_distorted.copy(), [target_contour_distorted], color_bgr=(0, 255, 0),
        )
        vis_img_distorted = self.draw_contours_on_image(
            vis_img_distorted, intersection_contours_distorted, color_bgr=(255, 255, 255), fill=True,
        )

        vis_img_undistorted = self.draw_contours_on_image(
            img_base_undistorted.copy(), [target_contour_undistorted], color_bgr=(0, 255, 0),
        )
        vis_img_undistorted = self.draw_contours_on_image(
            vis_img_undistorted, intersection_contours_undistorted, color_bgr=(255, 255, 255), fill=True,
        )

        return (
            iou_score,
            vis_img_distorted, obj_contour_distorted, target_contour_distorted,
            vis_img_undistorted, obj_contour_undistorted, target_contour_undistorted,
            alignment_angle_deg, alignment_warp,
        )

    def _detect_red_object_contour(self, img_bgr, apply_kmeans=True):
        mask = self.create_hsv_mask(
            img_bgr,
            self._RED_HSV_LOW,  self._RED_HSV_HIGH,
            self._RED_HSV_LOW2, self._RED_HSV_HIGH2,
        )
        largest_contour, _ = self.find_largest_contour(mask)
        if not apply_kmeans:
            return largest_contour
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        mask = self.k_means_darker_mask(img_bgr, mask)
        contour, _ = self.find_largest_contour(mask)
        return contour

    def _compose_euclidean_warps(self, warp_a, warp_b):
        m_a = np.vstack([warp_a, [0, 0, 1]])
        m_b = np.vstack([warp_b, [0, 0, 1]])
        return (m_a @ m_b)[:2].astype(np.float32)

    def _build_warp(self, cx_target, cy_target, cx_detected, cy_detected, angle_rad):
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        tx = cx_detected - cos_a * cx_target + sin_a * cy_target
        ty = cy_detected - sin_a * cx_target - cos_a * cy_target
        return np.array([[cos_a, -sin_a, tx],
                         [sin_a,  cos_a, ty]], dtype=np.float32)

    def _apply_warp_to_contour(self, contour, warp):
        pts = np.asarray(contour, dtype=np.float32)
        return cv2.transform(pts, warp).astype(np.int32)

    def _contour_iou(self, contour_a, contour_b, H, W):
        mask_a = np.zeros((H, W), dtype=np.uint8)
        mask_b = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask_a, [contour_a], -1, 255, -1)
        cv2.drawContours(mask_b, [contour_b], -1, 255, -1)
        intersection = np.count_nonzero(cv2.bitwise_and(mask_a, mask_b))
        union        = np.count_nonzero(cv2.bitwise_or(mask_a, mask_b))
        return intersection / union if union > 0 else 0.0
