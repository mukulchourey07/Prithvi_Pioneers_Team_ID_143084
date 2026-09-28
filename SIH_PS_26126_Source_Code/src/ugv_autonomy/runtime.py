"""The end-to-end camera -> perception -> VO -> replan -> motor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .control import PurePursuitController, SafetyMonitor, VehicleAdapter
from .mapping import GroundProjector, OccupancyGrid
from .models import DriveCommand, MotionEstimate, PerceptionResult
from .overtake import OvertakePlanner
from .perception import TerrainPerception
from .planning import AStarPlanner, PlanResult
from .slam import MonocularOrbSlam3
from .visual_odometry import VisualOdometry


@dataclass
class PipelineOutput:
    perception: PerceptionResult
    motion: MotionEstimate
    plan: PlanResult
    command: DriveCommand
    map_image: np.ndarray
    localisation_image: np.ndarray


class AutonomyRuntime:
    """Stateful autonomy stack; receives pixels and outputs a safe drive command."""

    def __init__(self, config: Dict[str, object], vehicle: VehicleAdapter | None = None):
        camera = config["camera"]
        self.goal: Tuple[float, float] = tuple(float(v) for v in config["goal"])
        self.perception = TerrainPerception(config["perception"], float(camera["horizon_fraction"]), camera)
        self.odometry = VisualOdometry(config["vo"])
        self.slam = None
        self.localisation_mode = "visual_odometry"
        self.slam_unavailable_reason = ""
        slam_config = config.get("slam", {})
        if bool(slam_config.get("enabled", False)):
            try:
                self.slam = MonocularOrbSlam3(slam_config)
                self.localisation_mode = "orbslam3_monocular"
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                # A missing native binding or vocabulary must never leave the
                # vehicle without localisation. Keep the proven VO path alive.
                self.slam_unavailable_reason = str(error)
                self.localisation_mode = "visual_odometry_fallback"
        self.projector = GroundProjector(camera, config["vo"])
        self.grid = OccupancyGrid(config["mapping"])
        self.planner = AStarPlanner(self.grid, config["mapping"])
        self.overtake = OvertakePlanner(config.get("overtake", {}))
        self.overtake_speed_scale = float(config.get("overtake", {}).get("speed_scale", 0.45))
        self.local_corridor_speed_scale = float(config.get("controller", {}).get("local_corridor_speed_scale", 0.50))
        self.controller = PurePursuitController(config["controller"])
        self.vehicle = vehicle or VehicleAdapter()
        distortion = camera.get("distortion_coefficients", [])
        self.distortion = np.asarray(distortion, dtype=np.float64) if distortion else None
        self.camera_matrix = np.array(
            [[float(config["vo"]["focal_length_px"]), 0.0, float(config["vo"]["principal_point_x"])],
             [0.0, float(config["vo"]["focal_length_px"]), float(config["vo"]["principal_point_y"])],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.frame_count = 0
        self.pose_history: list[Tuple[float, float]] = []

    def process(self, frame_bgr: np.ndarray) -> PipelineOutput:
        self.frame_count += 1
        if self.distortion is not None and self.distortion.size >= 4:
            frame_bgr = cv2.undistort(frame_bgr, self.camera_matrix, self.distortion)
        observed = self.perception.detect(frame_bgr)
        motion = self.slam.update(frame_bgr, time.monotonic()) if self.slam else self.odometry.update(frame_bgr)
        pose = motion.pose
        if motion.valid:
            self.pose_history.append((pose.x, pose.y))
        self.pose_history = self.pose_history[-2000:]
        self.grid.integrate_camera_masks(pose, self.projector, observed.drivable_mask, observed.obstacle_mask)
        standard_plan = self.planner.plan((pose.x, pose.y), self.goal)
        passing_plan = self.overtake.plan(pose, observed.hazards, self.goal, self.planner)
        plan = passing_plan if passing_plan is not None and passing_plan.status.startswith("overtake_") else standard_plan
        if not plan.path:
            local_path = self._local_safe_corridor(pose, observed)
            if local_path:
                plan = PlanResult(local_path, "local_safe_corridor")
        if plan.status.startswith("overtake_"):
            speed_scale = self.overtake_speed_scale
        elif plan.status == "local_safe_corridor":
            speed_scale = self.local_corridor_speed_scale
        else:
            speed_scale = 1.0
        command = self.controller.command(pose, plan.path, self.goal, speed_scale)
        # A central close obstacle remains an emergency stop unless a verified
        # free-ground overtake path was generated in this exact frame.
        if SafetyMonitor.immediate_hazard(observed.obstacle_mask) and not command.emergency_stop and not plan.status.startswith("overtake_"):
            command = DriveCommand(0.0, 0.0, True, "immediate_camera_hazard")
        sent = self.vehicle.send(command)
        annotated = self._add_status(observed.annotated, motion, plan, sent)
        observed.annotated = annotated
        map_image = self.grid.draw(pose, self.goal, plan.path)
        localisation_image = self._draw_localisation_trajectory(pose, motion)
        return PipelineOutput(observed, motion, plan, sent, map_image, localisation_image)

    def _local_safe_corridor(self, pose, observed: PerceptionResult) -> list[Tuple[float, float]]:
        """Create a short path solely through the latest confirmed safe pixels.

        The global goal can be outside the current camera range. If A* is
        temporarily blocked by conservative map evidence, this keeps the UGV
        moving slowly along its visible corridor, without treating unseen space
        as safe. A dilated obstacle mask creates a buffer around detections.
        """
        safe = observed.drivable_mask.copy()
        obstacle_buffer = cv2.dilate(observed.obstacle_mask, np.ones((17, 17), np.uint8))
        safe[obstacle_buffer > 0] = 0
        lateral, forward = self.projector.mask_points(safe, stride=4)
        if len(forward) == 0:
            return []
        goal_lateral, _ = pose.world_to_local(*self.goal)
        points: list[Tuple[float, float]] = []
        # Each band is a near-term waypoint. Select the point nearest the
        # desired heading but only from camera-confirmed, buffered safe ground.
        for distance in np.arange(1.2, min(self.projector.max_range - 0.4, 7.0), 1.0):
            candidates = np.flatnonzero(np.abs(forward - distance) <= 0.45)
            if len(candidates) == 0:
                continue
            selected = candidates[int(np.argmin(np.abs(lateral[candidates] - goal_lateral)))]
            point = pose.local_to_world(float(lateral[selected]), float(forward[selected]))
            if not points or np.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 0.35:
                points.append(point)
        return points if points and points[-1] != (pose.x, pose.y) else []

    @staticmethod
    def _add_status(image: np.ndarray, motion: MotionEstimate, plan: PlanResult, command: DriveCommand) -> np.ndarray:
        output = image.copy()
        status = f"VO: {motion.method} | features: {motion.tracked_features} | inliers: {motion.inliers}"
        route = f"Route: {plan.status} | steer: {command.steering:+.2f} | throttle: {command.throttle:.2f} | {command.reason}"
        cv2.rectangle(output, (0, output.shape[0] - 48), (output.shape[1], output.shape[0]), (20, 20, 20), -1)
        cv2.putText(output, status, (8, output.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        colour = (70, 240, 80) if not command.emergency_stop else (50, 80, 255)
        cv2.putText(output, route, (8, output.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1, cv2.LINE_AA)
        return output

    def _draw_localisation_trajectory(self, pose, motion: MotionEstimate) -> np.ndarray:
        """Debug display of the actual localiser pose trail, not a fake 3-D map."""
        canvas = np.full((520, 520, 3), 245, dtype=np.uint8)
        centre = np.array((260, 420), dtype=float)
        scale = 22.0
        trail = self.pose_history or [(pose.x, pose.y)]
        points = np.array([(centre[0] + x * scale, centre[1] - y * scale) for x, y in trail], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, (235, 90, 35), 2, cv2.LINE_AA)
        current = (int(centre[0] + pose.x * scale), int(centre[1] - pose.y * scale))
        cv2.circle(canvas, current, 7, (30, 80, 245), -1)
        end = (int(current[0] + 20 * np.sin(pose.yaw)), int(current[1] - 20 * np.cos(pose.yaw)))
        cv2.arrowedLine(canvas, current, end, (20, 20, 20), 2, cv2.LINE_AA, tipLength=0.28)
        mode = "ORB-SLAM3 MONOCULAR" if self.slam else "VISUAL ODOMETRY FALLBACK"
        state = "TRACKING" if motion.valid else "INITIALISING / TRACKING LOST"
        colour = (40, 145, 40) if self.slam and motion.valid else (30, 80, 220)
        cv2.putText(canvas, "GPS-free localisation trajectory", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(canvas, mode, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, state, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"x={pose.x:+.2f} m  y={pose.y:+.2f} m  yaw={np.degrees(pose.yaw):+.1f} deg", (12, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        return canvas
