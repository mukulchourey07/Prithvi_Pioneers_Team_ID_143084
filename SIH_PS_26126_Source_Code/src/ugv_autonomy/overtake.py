"""Conservative verified-space overtaking planner."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .models import Hazard, Pose2D
from .planning import AStarPlanner, PlanResult


class OvertakePlanner:
    """Plan a side-pass only through recently observed free ground.

    This is intentionally not an assertive road-traffic overtaking policy. It
    only creates a low-speed UGV side-pass around a static/slow obstacle when a
    robot-width-safe detour is confirmed by the local drivable-area map.
    """

    def __init__(self, config: Dict[str, object]):
        self.enabled = bool(config.get("enabled", True))
        self.trigger_distance = float(config.get("trigger_distance_m", 7.0))
        self.path_half_width = float(config.get("path_half_width_m", 0.85))
        self.lateral_offset = float(config.get("lateral_offset_m", 1.25))
        self.clear_ahead = float(config.get("clear_ahead_m", 1.4))
        self.allowed_labels = set(config.get("allowed_labels", ["person", "vehicle", "bicycle", "animal", "obstacle"]))

    def plan(
        self,
        pose: Pose2D,
        hazards: Iterable[Hazard],
        goal: Tuple[float, float],
        planner: AStarPlanner,
    ) -> PlanResult | None:
        if not self.enabled:
            return None
        candidate = self._blocking_hazard(hazards)
        if candidate is None:
            return None
        lateral, forward = candidate.ground_point  # checked by _blocking_hazard
        # Prefer the side farther from an obstacle that is already offset from
        # the UGV's centreline. Both candidates require observed free space.
        sides = (-1.0, 1.0) if lateral >= 0 else (1.0, -1.0)
        for side in sides:
            waypoint = pose.local_to_world(side * self.lateral_offset, forward + self.clear_ahead)
            approach = planner.plan((pose.x, pose.y), waypoint, verified_free_only=True)
            if approach.status != "ok" or len(approach.path) < 2:
                continue
            onward = planner.plan(waypoint, goal)
            if onward.status != "ok":
                continue
            direction = "left" if side < 0 else "right"
            return PlanResult(approach.path + onward.path[1:], f"overtake_{direction}")
        return PlanResult([], "overtake_blocked")

    def _blocking_hazard(self, hazards: Iterable[Hazard]) -> Hazard | None:
        candidates = []
        for hazard in hazards:
            if hazard.label not in self.allowed_labels or hazard.ground_point is None:
                continue
            lateral, forward = hazard.ground_point
            if 0.7 <= forward <= self.trigger_distance and abs(lateral) <= self.path_half_width:
                candidates.append(hazard)
        return min(candidates, key=lambda hazard: hazard.ground_point[1]) if candidates else None
