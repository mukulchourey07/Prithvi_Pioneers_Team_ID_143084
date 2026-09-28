"""Projection of camera masks onto a local, GPS-free occupancy map."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .models import Pose2D


class GroundProjector:
    """Approximate image-to-ground projection for a calibrated forward camera.

    A flat-ground inverse perspective mapping is appropriate for short-range
    obstacle avoidance. Replace this with calibrated homography/stereo depth for
    rugged terrain or large camera pitch changes.
    """

    def __init__(self, camera_config: Dict[str, object], vo_config: Dict[str, object] | None = None):
        self.width = int(camera_config.get("width", 640))
        self.height = int(camera_config.get("height", 480))
        self.horizon_fraction = float(camera_config.get("horizon_fraction", 0.25))
        self.horizon = int(self.height * self.horizon_fraction)
        self.max_range = float(camera_config.get("max_ground_range_m", 12.0))
        self.camera_height = float(camera_config.get("camera_height_m", 0.0))
        self.half_fov_tan = np.tan(np.deg2rad(float(camera_config.get("horizontal_fov_degrees", 76.0)) / 2.0))
        vo = vo_config or {}
        self.focal = float(vo.get("focal_length_px", 0.0))
        self.cx = float(vo.get("principal_point_x", self.width / 2))

    def pixel_to_ground(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return lateral and forward metres for pixels below the horizon."""
        current_h, current_w = self.height, self.width
        pixels_u, pixels_v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        horizon = current_h * self.horizon_fraction
        if self.camera_height > 0.0 and self.focal > 0.0:
            # Flat-ground pinhole projection. This uses the measured physical
            # camera height and calibrated focal length for metric obstacle maps.
            forward = self.camera_height * self.focal / np.maximum(1.0, pixels_v - horizon)
            lateral = (pixels_u - self.cx) * forward / self.focal
        else:
            forward = self.max_range * (current_h - pixels_v) / max(1, current_h - horizon)
            lateral = ((pixels_u - current_w / 2) / (current_w / 2)) * forward * self.half_fov_tan
        forward = np.clip(forward, 0.35, self.max_range)
        return lateral, forward

    def mask_points(self, mask: np.ndarray, stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(mask[self.horizon :, :] > 0)
        ys = ys + self.horizon
        if len(xs) == 0:
            return np.empty(0), np.empty(0)
        selection = np.arange(0, len(xs), stride)
        return self.pixel_to_ground(xs[selection], ys[selection])


class OccupancyGrid:
    """Log-odds local world map with no global/GPS reference."""

    def __init__(self, config: Dict[str, object]):
        self.resolution = float(config.get("resolution_m", 0.20))
        self.width_m = float(config.get("width_m", 30.0))
        self.height_m = float(config.get("height_m", 30.0))
        self.origin_x = float(config.get("origin_x_m", -15.0))
        self.origin_y = float(config.get("origin_y_m", -2.0))
        self.width = int(round(self.width_m / self.resolution))
        self.height = int(round(self.height_m / self.resolution))
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        # A side-pass is allowed only where the camera has actually confirmed
        # free ground; unknown cells remain available to ordinary long-range
        # navigation but not to the overtaking manoeuvre.
        self.observed_free = np.zeros((self.height, self.width), dtype=bool)

    def clear(self) -> None:
        self.log_odds.fill(0)
        self.observed_free.fill(False)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int] | None:
        col = int(np.floor((x - self.origin_x) / self.resolution))
        row_from_bottom = int(np.floor((y - self.origin_y) / self.resolution))
        row = self.height - 1 - row_from_bottom
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (self.height - row - 0.5) * self.resolution
        return x, y

    def integrate_camera_masks(
        self,
        pose: Pose2D,
        projector: GroundProjector,
        drivable_mask: np.ndarray,
        obstacle_mask: np.ndarray,
    ) -> None:
        """Fuse the latest terrain observations using the visual-odometry pose."""
        # Free evidence is sparse and weak. It allows a route to reopen after a
        # transient segmentation error without erasing a repeated hazard quickly.
        for lateral, forward in zip(*projector.mask_points(drivable_mask, stride=8)):
            if forward < 1.0:
                continue
            cell = self.world_to_cell(*pose.local_to_world(float(lateral), float(forward)))
            if cell is not None:
                self.log_odds[cell] = max(-3.5, self.log_odds[cell] - 0.16)
                self.observed_free[cell] = True
        for lateral, forward in zip(*projector.mask_points(obstacle_mask, stride=6)):
            cell = self.world_to_cell(*pose.local_to_world(float(lateral), float(forward)))
            if cell is not None:
                self.log_odds[cell] = min(5.0, self.log_odds[cell] + 0.75)
                self.observed_free[cell] = False

    def occupied(self, probability: float = 0.62) -> np.ndarray:
        threshold = float(np.log(probability / (1.0 - probability)))
        return self.log_odds >= threshold

    def draw(
        self,
        pose: Pose2D,
        goal: Tuple[float, float],
        path: Iterable[Tuple[float, float]] = (),
        scale: int = 4,
    ) -> np.ndarray:
        """Return a north-up debug image of map state and planned route."""
        probability = 1.0 / (1.0 + np.exp(-self.log_odds))
        canvas = np.full((self.height, self.width, 3), 235, dtype=np.uint8)
        canvas[probability < 0.35] = (175, 215, 175)  # observed free
        canvas[probability > 0.62] = (35, 35, 35)  # occupied
        canvas = cv2.resize(canvas, (self.width * scale, self.height * scale), interpolation=cv2.INTER_NEAREST)
        points = [self.world_to_cell(x, y) for x, y in path]
        points = [point for point in points if point is not None]
        if len(points) > 1:
            cv2.polylines(canvas, [np.array([(c * scale + scale // 2, r * scale + scale // 2) for r, c in points], np.int32)], False, (235, 70, 40), 2)
        for point, colour, radius in ((self.world_to_cell(*goal), (255, 50, 210), 5), (self.world_to_cell(pose.x, pose.y), (30, 80, 245), 5)):
            if point is not None:
                r, c = point
                cv2.circle(canvas, (c * scale + scale // 2, r * scale + scale // 2), radius, colour, -1)
        cv2.putText(canvas, "local visual-odometry map (north/up)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        return canvas
