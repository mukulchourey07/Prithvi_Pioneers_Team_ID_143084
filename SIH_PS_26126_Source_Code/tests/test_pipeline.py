"""Fast regression tests for map planning, perception, and command safety."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ugv_autonomy.control import PurePursuitController
from ugv_autonomy.mapping import OccupancyGrid
from ugv_autonomy.models import Hazard, Pose2D
from ugv_autonomy.overtake import OvertakePlanner
from ugv_autonomy.perception import TerrainPerception
from ugv_autonomy.planning import AStarPlanner
from ugv_autonomy.planning import PlanResult
from ugv_autonomy.runtime import AutonomyRuntime
from ugv_autonomy.simulator import UGVSimulator


class PipelineTests(unittest.TestCase):
    def test_simulated_safe_ground_and_hazards_are_perceived(self):
        simulator = UGVSimulator({"width": 640, "height": 480, "horizon_fraction": 0.25, "max_ground_range_m": 12, "horizontal_fov_degrees": 76})
        detector = TerrainPerception({"minimum_obstacle_area_px": 80}, 0.25)
        output = detector.detect(simulator.frame())
        # The safe-ground replacement only marks the connected road corridor
        # drivable; unverified terrain is obstacle evidence for the map.
        self.assertGreater(output.drivable_mask.sum(), 1000)
        self.assertGreater(output.obstacle_mask.sum(), 1000)
        self.assertEqual(output.drivable_mask.shape, output.obstacle_mask.shape)
        self.assertIn("tree", {hazard.label for hazard in output.hazards})

    def test_overtake_requires_observed_free_detour(self):
        grid = OccupancyGrid({"resolution_m": 0.25, "width_m": 12, "height_m": 14, "origin_x_m": -6, "origin_y_m": -1})
        # Confirmed drivable space lies only to the left of a central obstacle.
        for x in np.arange(-2.0, 0.1, 0.25):
            for y in np.arange(0.0, 7.1, 0.25):
                cell = grid.world_to_cell(float(x), float(y))
                grid.observed_free[cell] = True
        planner = AStarPlanner(grid, {"robot_radius_m": 0.35})
        overtake = OvertakePlanner({"enabled": True, "lateral_offset_m": 1.0, "clear_ahead_m": 1.0})
        hazard = Hazard("vehicle", 0.95, (300, 270, 40, 80), (0.0, 3.0))
        plan = overtake.plan(Pose2D(), [hazard], (0.0, 8.0), planner)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.status, "overtake_left")

    def test_local_safe_corridor_replaces_transient_blocked_global_plan(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        runtime = AutonomyRuntime(config)
        runtime.planner.plan = lambda *_args, **_kwargs: PlanResult([], "route_blocked")
        output = runtime.process(UGVSimulator(config["camera"]).frame())
        self.assertEqual(output.plan.status, "local_safe_corridor")
        self.assertFalse(output.command.emergency_stop)
        self.assertGreater(output.command.throttle, 0.0)

    def test_a_star_routes_around_obstacle(self):
        grid = OccupancyGrid({"resolution_m": 0.25, "width_m": 12, "height_m": 14, "origin_x_m": -6, "origin_y_m": -1})
        for x in (-0.25, 0.0, 0.25):
            for y in (3.75, 4.0, 4.25):
                cell = grid.world_to_cell(x, y)
                grid.log_odds[cell] = 5.0
        plan = AStarPlanner(grid, {"robot_radius_m": 0.35}).plan((0.0, 0.0), (0.0, 8.0))
        self.assertEqual(plan.status, "ok")
        self.assertGreater(len(plan.path), 2)
        self.assertTrue(any(abs(x) > 0.4 for x, _ in plan.path))

    def test_controller_stops_without_route(self):
        controller = PurePursuitController({})
        command = controller.command(Pose2D(), [], (0.0, 5.0))
        self.assertTrue(command.emergency_stop)
        self.assertEqual(command.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
