"""A deterministic virtual UGV and forward camera used for safe end-to-end demos.

Its internal pose is never given to the autonomy modules. It only renders the
camera pixels and applies the abstract motor command, exactly as hardware would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from .models import DriveCommand, Pose2D


@dataclass
class WorldHazard:
    kind: str
    x: float
    y: float
    radius: float
    colour: tuple[int, int, int]


class UGVSimulator:
    def __init__(self, camera_config: Dict[str, object]):
        self.width = int(camera_config.get("width", 640))
        self.height = int(camera_config.get("height", 480))
        self.horizon = int(self.height * float(camera_config.get("horizon_fraction", 0.25)))
        self.max_range = float(camera_config.get("max_ground_range_m", 12.0))
        self.half_fov_tan = np.tan(np.deg2rad(float(camera_config.get("horizontal_fov_degrees", 76.0)) / 2.0))
        self.pose = Pose2D()
        self.step_count = 0
        self.collided = False
        self.hazards: List[WorldHazard] = [
            WorldHazard("tree", 0.0, 6.3, 0.78, (45, 135, 35)),
            WorldHazard("rock", 2.4, 10.1, 0.62, (115, 115, 115)),
            WorldHazard("person", -2.35, 12.8, 0.42, (220, 65, 25)),
            WorldHazard("ditch", -3.4, 9.0, 0.90, (28, 28, 28)),
        ]
        rng = np.random.default_rng(2026)
        self.texture_points = np.column_stack((rng.uniform(-14, 14, 1500), rng.uniform(-2, 30, 1500), rng.integers(0, 4, 1500)))

    def frame(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), (220, 195, 150), dtype=np.uint8)
        frame[: self.horizon] = (205, 218, 225)  # sky
        frame[self.horizon :] = (68, 120, 180)  # warm earth, within ground HSV configuration
        # Stable terrain texture supplies image features for visual odometry.
        for x, y, tone in self.texture_points:
            lateral, forward = self.pose.world_to_local(float(x), float(y))
            projected = self._project(lateral, forward)
            if projected is None:
                continue
            u, v = projected
            colour = ((58, 100, 155), (76, 132, 190), (52, 108, 166), (84, 140, 198))[int(tone)]
            radius = 1 if forward > 4 else 2
            cv2.circle(frame, (u, v), radius, colour, -1)
        for hazard in self.hazards:
            lateral, forward = self.pose.world_to_local(hazard.x, hazard.y)
            projected = self._project(lateral, forward)
            if projected is None:
                continue
            u, v = projected
            if hazard.kind == "ditch":
                half_width = max(12, int(105 / max(forward, 1.0)))
                thickness = max(5, int(32 / max(forward, 1.0)))
                cv2.rectangle(frame, (u - half_width, v - thickness), (u + half_width, v + thickness), hazard.colour, -1)
                cv2.line(frame, (u - half_width, v - thickness), (u + half_width, v - thickness), (70, 70, 70), 1)
            elif hazard.kind == "tree":
                canopy = max(11, int(128 / max(forward, 1.0)))
                cv2.rectangle(frame, (u - canopy // 5, v - canopy), (u + canopy // 5, v + canopy // 4), (35, 80, 75), -1)
                cv2.circle(frame, (u, v - canopy // 2), canopy, hazard.colour, -1)
            elif hazard.kind == "person":
                person_h = max(18, int(145 / max(forward, 1.0)))
                cv2.circle(frame, (u, v - person_h // 2), max(4, person_h // 7), (180, 170, 150), -1)
                cv2.rectangle(frame, (u - person_h // 6, v - person_h // 3), (u + person_h // 6, v + person_h // 2), hazard.colour, -1)
            else:
                radius = max(8, int(115 * hazard.radius / max(forward, 1.0)))
                cv2.circle(frame, (u, v), radius, hazard.colour, -1)
                cv2.circle(frame, (u - radius // 3, v - radius // 3), max(2, radius // 4), (170, 170, 170), -1)
        cv2.putText(frame, "SIMULATED FRONT CAMERA -- NO GPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 45), 2, cv2.LINE_AA)
        cv2.putText(frame, "SIMULATED FRONT CAMERA -- NO GPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 1, cv2.LINE_AA)
        return frame

    def apply(self, command: DriveCommand, dt: float = 0.12) -> None:
        self.step_count += 1
        if command.emergency_stop:
            return
        speed = 1.05 * command.throttle
        yaw = self.pose.yaw + float(command.steering) * 0.92 * dt
        proposed = Pose2D(self.pose.x + speed * np.sin(yaw) * dt, self.pose.y + speed * np.cos(yaw) * dt, yaw)
        if any(np.hypot(proposed.x - hazard.x, proposed.y - hazard.y) < (hazard.radius + 0.42) for hazard in self.hazards):
            self.collided = True
            return
        self.pose = proposed
        # New obstacle demonstrates route updates after the initial plan.
        if self.step_count == 95:
            self.hazards.append(WorldHazard("rock", 1.15, 14.6, 0.52, (100, 100, 100)))

    def _project(self, lateral: float, forward: float) -> tuple[int, int] | None:
        if forward < 0.35 or forward > self.max_range:
            return None
        if abs(lateral) > forward * self.half_fov_tan:
            return None
        u = int(self.width / 2 + lateral / (forward * self.half_fov_tan) * self.width / 2)
        v = int(self.height - forward / self.max_range * (self.height - self.horizon))
        if not (0 <= u < self.width and self.horizon <= v < self.height):
            return None
        return u, v
