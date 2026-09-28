"""Route following and a deliberately hardware-agnostic drive-command interface."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Tuple

import numpy as np

from .models import DriveCommand, Pose2D


class PurePursuitController:
    def __init__(self, config: Dict[str, object]):
        self.lookahead = float(config.get("lookahead_m", 1.3))
        self.max_steering = float(config.get("max_steering_rad", 0.55))
        self.cruise_throttle = float(config.get("cruise_throttle", 0.42))
        self.arrival_radius = float(config.get("arrival_radius_m", 0.55))

    def command(self, pose: Pose2D, path: Iterable[Tuple[float, float]], goal: Tuple[float, float], speed_scale: float = 1.0) -> DriveCommand:
        route = list(path)
        if not route:
            return DriveCommand(0.0, 0.0, True, "no_safe_route")
        if np.hypot(goal[0] - pose.x, goal[1] - pose.y) <= self.arrival_radius:
            return DriveCommand(0.0, 0.0, True, "destination_reached")
        target = route[-1]
        for point in route:
            if np.hypot(point[0] - pose.x, point[1] - pose.y) >= self.lookahead:
                target = point
                break
        lateral, forward = pose.world_to_local(*target)
        heading_error = float(np.arctan2(lateral, max(0.05, forward)))
        steering_rad = float(np.clip(heading_error, -self.max_steering, self.max_steering))
        # Slow before sharp turns; always leave enough drive to clear a local turn.
        turn_ratio = abs(steering_rad) / self.max_steering
        throttle = self.cruise_throttle * (1.0 - 0.48 * turn_ratio) * float(np.clip(speed_scale, 0.0, 1.0))
        return DriveCommand(steering_rad / self.max_steering, throttle, False, "pure_pursuit")


class VehicleAdapter:
    """Boundary for sending validated commands to a real UGV controller.

    The base implementation stores the command only. Implement `send` with a
    verified serial/CAN protocol and independent hardware E-stop before use on a
    vehicle. Runtime never embeds a device-specific motor command.
    """

    def __init__(self):
        self.last_command = DriveCommand(0.0, 0.0, True, "startup")

    def send(self, command: DriveCommand) -> DriveCommand:
        steering = float(np.clip(command.steering, -1.0, 1.0))
        throttle = float(np.clip(command.throttle, 0.0, 1.0))
        if command.emergency_stop:
            steering, throttle = 0.0, 0.0
        self.last_command = DriveCommand(steering, throttle, command.emergency_stop, command.reason)
        return self.last_command

    def close(self) -> None:
        """Release optional hardware resources; the base adapter has none."""


class JsonSerialVehicleAdapter(VehicleAdapter):
    """Explicitly armed serial bridge for normalized steering/motor commands.

    The UGV microcontroller must accept one UTF-8 JSON object per line with
    ``steering`` in [-1, 1], ``throttle`` in [0, 1], and ``emergency_stop``.
    The bridge is intentionally disarmed unless the operator changes the
    configuration after a bench test with drive wheels off the ground.
    """

    def __init__(self, config: Dict[str, object]):
        super().__init__()
        self.armed = bool(config.get("armed", False))
        self.serial = None
        self.status = "disarmed"
        if not self.armed:
            return
        try:
            import serial

            self.serial = serial.Serial(
                port=str(config["port"]),
                baudrate=int(config.get("baudrate", 115200)),
                timeout=float(config.get("timeout_s", 0.1)),
                write_timeout=float(config.get("timeout_s", 0.1)),
            )
            self.status = "armed"
        except (ImportError, KeyError, OSError, ValueError):
            self.armed = False
            self.status = "serial_unavailable"

    def send(self, command: DriveCommand) -> DriveCommand:
        safe_command = super().send(command)
        if self.serial is None:
            return safe_command
        payload = {
            "steering": round(safe_command.steering, 4),
            "throttle": round(safe_command.throttle, 4),
            "emergency_stop": safe_command.emergency_stop,
        }
        try:
            self.serial.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        except OSError:
            self.last_command = DriveCommand(0.0, 0.0, True, "serial_write_failed")
        return self.last_command

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


class SafetyMonitor:
    """Last-line visual stop zone in front of the bumper."""

    @staticmethod
    def immediate_hazard(obstacle_mask: np.ndarray) -> bool:
        h, w = obstacle_mask.shape[:2]
        # Closest central 1.2--2.5 m equivalent. Ignore bottom border noise.
        zone = obstacle_mask[int(h * 0.68) : int(h * 0.91), int(w * 0.40) : int(w * 0.60)]
        return zone.size > 0 and float(np.count_nonzero(zone)) / zone.size > 0.10
