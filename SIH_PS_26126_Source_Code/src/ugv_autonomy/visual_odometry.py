"""Monocular, feature-based visual odometry.

This module intentionally has no GPS or simulator-pose dependency. It estimates
inter-frame camera motion with ORB features, an essential matrix, and RANSAC;
the resulting trajectory is in a local frame starting at (0, 0). Monocular
translation requires a separately calibrated scale prior.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .models import MotionEstimate, Pose2D


class VisualOdometry:
    def __init__(self, config: Dict[str, object]):
        self.focal = float(config.get("focal_length_px", 520.0))
        self.cx = float(config.get("principal_point_x", 320.0))
        self.cy = float(config.get("principal_point_y", 240.0))
        self.scale = float(config.get("vo_scale_m_per_frame", 0.10))
        self.min_inliers = int(config.get("min_inliers", 12))
        self.orb = cv2.ORB_create(nfeatures=int(config.get("max_features", 1200)), fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_gray: np.ndarray | None = None
        self.prev_kp = None
        self.prev_desc: np.ndarray | None = None
        self.pose = Pose2D()

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]], dtype=np.float64)

    def update(self, frame_bgr: np.ndarray) -> MotionEstimate:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if self.prev_desc is None or descriptors is None or len(keypoints) < 8:
            self._remember(gray, keypoints, descriptors)
            return MotionEstimate(self.pose, len(keypoints), 0, False, "initialising")

        raw_matches = self.matcher.knnMatch(self.prev_desc, descriptors, k=2)
        good = [pair[0] for pair in raw_matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if len(good) < 8:
            self._remember(gray, keypoints, descriptors)
            return MotionEstimate(self.pose, len(good), 0, False, "not_enough_matches")

        previous_points = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        current_points = np.float32([keypoints[m.trainIdx].pt for m in good])
        essential, mask = cv2.findEssentialMat(
            previous_points,
            current_points,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.2,
        )
        if essential is None or mask is None:
            self._remember(gray, keypoints, descriptors)
            return MotionEstimate(self.pose, len(good), 0, False, "essential_matrix_failed")
        try:
            inliers, rotation, translation, _ = cv2.recoverPose(
                essential, previous_points, current_points, self.camera_matrix, mask=mask
            )
        except cv2.error:
            self._remember(gray, keypoints, descriptors)
            return MotionEstimate(self.pose, len(good), 0, False, "pose_recovery_failed")

        self._remember(gray, keypoints, descriptors)
        if inliers < self.min_inliers:
            return MotionEstimate(self.pose, len(good), int(inliers), False, "insufficient_ransac_inliers")

        # `recoverPose` gives X_current = R X_previous + t. Camera-centre motion
        # in the previous camera frame is -R.T @ t. The optical camera z-axis is
        # vehicle forward and x is vehicle right. Sign is disambiguated by RANSAC
        # chirality; its magnitude is set by a field-calibrated monocular scale.
        camera_motion = -rotation.T @ translation.reshape(3)
        lateral = float(camera_motion[0] * self.scale)
        forward = float(camera_motion[2] * self.scale)
        yaw_delta = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        # Reject numerically possible but physically implausible one-frame jumps.
        if abs(lateral) > 0.8 or abs(forward) > 0.8 or abs(yaw_delta) > 0.5:
            return MotionEstimate(self.pose, len(good), int(inliers), False, "implausible_motion_rejected")
        self.pose = self.pose.advance(lateral, forward, yaw_delta)
        return MotionEstimate(self.pose, len(good), int(inliers), True, "orb_essential_matrix")

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.pose = Pose2D()

    def _remember(self, gray: np.ndarray, keypoints, descriptors: np.ndarray | None) -> None:
        self.prev_gray = gray
        self.prev_kp = keypoints
        self.prev_desc = descriptors
