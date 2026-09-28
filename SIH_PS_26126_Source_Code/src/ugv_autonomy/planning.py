"""Inflated-occupancy A* global/local route planning."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .mapping import OccupancyGrid


@dataclass
class PlanResult:
    path: List[Tuple[float, float]]
    status: str


class AStarPlanner:
    def __init__(self, grid: OccupancyGrid, config: Dict[str, object]):
        self.grid = grid
        self.robot_radius = float(config.get("robot_radius_m", 0.50))

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], verified_free_only: bool = False) -> PlanResult:
        start_cell, goal_cell = self.grid.world_to_cell(*start), self.grid.world_to_cell(*goal)
        if start_cell is None or goal_cell is None:
            return PlanResult([], "start_or_goal_outside_local_map")
        blocked = self._inflated_obstacles(verified_free_only)
        start_cell = self._nearest_free(start_cell, blocked)
        goal_cell = self._nearest_free(goal_cell, blocked)
        if start_cell is None or goal_cell is None:
            return PlanResult([], "no_free_start_or_goal")

        frontier = [(0.0, start_cell)]
        came_from = {start_cell: None}
        cost = {start_cell: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                cells = []
                while current is not None:
                    cells.append(current)
                    current = came_from[current]
                cells.reverse()
                return PlanResult(self._simplify([self.grid.cell_to_world(*cell) for cell in cells]), "ok")
            for neighbour, step_cost in self._neighbours(current, blocked):
                new_cost = cost[current] + step_cost
                if new_cost < cost.get(neighbour, float("inf")):
                    cost[neighbour] = new_cost
                    priority = new_cost + _heuristic(neighbour, goal_cell)
                    heapq.heappush(frontier, (priority, neighbour))
                    came_from[neighbour] = current
        return PlanResult([], "route_blocked")

    def _inflated_obstacles(self, verified_free_only: bool = False) -> np.ndarray:
        occupied = self.grid.occupied().astype(np.uint8)
        radius_cells = max(1, int(np.ceil(self.robot_radius / self.grid.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_cells * 2 + 1, radius_cells * 2 + 1))
        blocked = cv2.dilate(occupied, kernel).astype(bool)
        if verified_free_only:
            # Dilate observed-free samples so sub-cell camera sampling does not
            # artificially break an otherwise observed traversable corridor.
            known_free = cv2.dilate(self.grid.observed_free.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            blocked |= ~known_free
        return blocked

    def _nearest_free(self, source: Tuple[int, int], blocked: np.ndarray) -> Tuple[int, int] | None:
        if not blocked[source]:
            return source
        for radius in range(1, 12):
            r0, c0 = source
            candidates = [(r, c) for r in range(r0 - radius, r0 + radius + 1) for c in range(c0 - radius, c0 + radius + 1) if 0 <= r < blocked.shape[0] and 0 <= c < blocked.shape[1] and not blocked[r, c]]
            if candidates:
                return min(candidates, key=lambda cell: _heuristic(cell, source))
        return None

    @staticmethod
    def _neighbours(cell: Tuple[int, int], blocked: np.ndarray):
        row, col = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1] and not blocked[nr, nc]:
                # Prevent diagonal corner-cutting between two blocked cells.
                if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                    continue
                yield (nr, nc), 1.414 if dr and dc else 1.0

    @staticmethod
    def _simplify(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        previous_direction = None
        for a, b in zip(path, path[1:]):
            direction = (round(b[0] - a[0], 3), round(b[1] - a[1], 3))
            normalized = (np.sign(direction[0]), np.sign(direction[1]))
            if previous_direction is not None and normalized != previous_direction:
                simplified.append(a)
            previous_direction = normalized
        simplified.append(path[-1])
        return simplified


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
