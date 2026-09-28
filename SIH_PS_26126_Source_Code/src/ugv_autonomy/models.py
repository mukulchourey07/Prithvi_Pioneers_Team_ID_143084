"""Small, dependency-free data models shared by the autonomy pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Pose2D:
    """Local-map pose: x is right/east, y is forward/north, yaw 0 faces +y."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def local_to_world(self, lateral_m: float, forward_m: float) -> Tuple[float, float]:
        """Transform a vehicle-frame ground point into the local map frame."""
        return (
            self.x + np.cos(self.yaw) * lateral_m + np.sin(self.yaw) * forward_m,
            self.y - np.sin(self.yaw) * lateral_m + np.cos(self.yaw) * forward_m,
        )

    def world_to_local(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a local-map point into the current vehicle frame."""
        dx, dy = x - self.x, y - self.y
        return (
            np.cos(self.yaw) * dx - np.sin(self.yaw) * dy,
            np.sin(self.yaw) * dx + np.cos(self.yaw) * dy,
        )

    def advance(self, lateral_m: float, forward_m: float, yaw_delta: float) -> "Pose2D":
        x, y = self.local_to_world(lateral_m, forward_m)
        return Pose2D(x=x, y=y, yaw=_wrap_angle(self.yaw + yaw_delta))


@dataclass
class Hazard:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    ground_point: Tuple[float, float] | None = None


@dataclass
class PerceptionResult:
    drivable_mask: np.ndarray
    obstacle_mask: np.ndarray
    annotated: np.ndarray
    hazards: List[Hazard] = field(default_factory=list)


@dataclass
class MotionEstimate:
    pose: Pose2D
    tracked_features: int
    inliers: int
    valid: bool
    method: str


@dataclass
class DriveCommand:
    steering: float
    throttle: float
    emergency_stop: bool = False
    reason: str = ""


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi
