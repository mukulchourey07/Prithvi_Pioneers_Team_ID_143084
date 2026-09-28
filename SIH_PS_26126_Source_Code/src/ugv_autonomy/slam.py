"""Optional ORB-SLAM3 monocular adapter for GPS-free localisation and loop closure.

The native ORB-SLAM3 library supplies keyframes, map points, relocalisation, and
loop closure. This module deliberately does not reimplement SLAM in Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .models import MotionEstimate, Pose2D


class MonocularOrbSlam3:
    """Adapt a supported `orbslam3-python` binding to the UGV Pose2D interface.

    Camera must be rigidly mounted and level. The binding reports monocular pose
    in arbitrary units; `metric_scale_m_per_slam_unit` must be calibrated by a
    measured straight-line drive. Camera height gives metric ground projection
    for obstacle placement, but cannot by itself remove monocular SLAM scale
    ambiguity unless a ground-plane scale estimator is also added.
    """

    def __init__(self, settings: Dict[str, object]):
        vocabulary = Path(str(settings.get("vocabulary_path", "")))
        camera_settings = Path(str(settings.get("settings_path", "")))
        if not vocabulary.is_file() or not camera_settings.is_file():
            raise FileNotFoundError("ORB-SLAM3 requires existing vocabulary_path and settings_path files")
        try:
            import orbslam3
        except ImportError as error:
            raise RuntimeError("Install a compatible ORB-SLAM3 Python binding: pip install orbslam3-python") from error
        self.binding = orbslam3
        self.scale = float(settings.get("metric_scale_m_per_slam_unit", 1.0))
        if self.scale <= 0:
            raise ValueError("metric_scale_m_per_slam_unit must be positive")
        self.pose_convention = str(settings.get("pose_convention", "Tcw"))
        self.system = self._create_system(str(vocabulary), str(camera_settings))
        self.origin: np.ndarray | None = None
        self.last_pose = Pose2D()

    def _create_system(self, vocabulary: str, settings: str):
        sensor = getattr(getattr(self.binding, "Sensor", object()), "MONOCULAR", None)
        if sensor is None:
            sensor = getattr(getattr(self.binding, "System", object()), "MONOCULAR", None)
        if sensor is None:
            raise RuntimeError("Installed ORB-SLAM3 binding does not expose MONOCULAR sensor mode")
        try:
            return self.binding.System(vocabulary, settings, sensor, False)
        except (AttributeError, TypeError) as error:
            raise RuntimeError("Unsupported ORB-SLAM3 binding API; expected System(vocabulary, settings, MONOCULAR, viewer)") from error

    def update(self, frame_bgr: np.ndarray, timestamp_s: float) -> MotionEstimate:
        track = getattr(self.system, "TrackMonocular", None) or getattr(self.system, "track_monocular", None)
        if track is None:
            return MotionEstimate(self.last_pose, 0, 0, False, "orbslam3_binding_has_no_monocular_tracker")
        try:
            returned_pose = track(frame_bgr, float(timestamp_s))
            pose_matrix = self._pose_matrix(returned_pose)
            if pose_matrix is None:
                get_pose = getattr(self.system, "get_frame_pose", None) or getattr(self.system, "GetFramePose", None)
                pose_matrix = self._pose_matrix(get_pose()) if get_pose else None
        except Exception:
            return MotionEstimate(self.last_pose, 0, 0, False, "orbslam3_tracking_lost")
        if pose_matrix is None or pose_matrix.shape != (4, 4) or not np.isfinite(pose_matrix).all():
            return MotionEstimate(self.last_pose, 0, 0, False, "orbslam3_initialising_or_lost")
        twc = np.linalg.inv(pose_matrix) if self.pose_convention.upper() == "TCW" else pose_matrix
        position = twc[:3, 3]
        if self.origin is None:
            self.origin = position.copy()
        relative = (position - self.origin) * self.scale
        rotation = twc[:3, :3]
        # Assumes a level, forward-facing camera: ORB world x maps to UGV right
        # and ORB world z maps to UGV forward after initialisation.
        yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        self.last_pose = Pose2D(float(relative[0]), float(relative[2]), yaw)
        return MotionEstimate(self.last_pose, 0, 0, True, "orbslam3_monocular")

    @staticmethod
    def _pose_matrix(candidate) -> np.ndarray | None:
        if candidate is None:
            return None
        matrix = np.asarray(candidate, dtype=np.float64)
        return matrix if matrix.shape == (4, 4) else None

    def shutdown(self) -> None:
        shutdown = getattr(self.system, "Shutdown", None) or getattr(self.system, "shutdown", None)
        if shutdown:
            shutdown()
