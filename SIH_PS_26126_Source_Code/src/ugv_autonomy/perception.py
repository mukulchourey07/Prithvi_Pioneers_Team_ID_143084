"""Safe-ground camera perception adapted for the autonomy pipeline.

The module owns only the camera-to-mask boundary. It preserves the
``PerceptionResult`` contract consumed by mapping, planning, and control.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .models import Hazard, PerceptionResult
from .yolop import YOLOPAdapter


# Semantic classes carried over from ugv_safe_ground_perception.
BACKGROUND = 0
TRAVERSABLE_PATH = 1
ROUGH_TERRAIN = 2
POSITIVE_OBSTACLE = 3
NEGATIVE_OBSTACLE = 4

CLASS_COLOURS_BGR = {
    TRAVERSABLE_PATH: (50, 205, 50),
    ROUGH_TERRAIN: (0, 165, 255),
    POSITIVE_OBSTACLE: (30, 30, 220),
    NEGATIVE_OBSTACLE: (210, 50, 180),
}


class TerrainPerception:
    """Adaptive safe-corridor segmentation and optional YOLO road hazards.

    A temporal sample of the ground immediately in front of the vehicle is
    used for unpaved terrain; neutral, illumination-invariant matching handles
    asphalt and concrete. Only the connected bottom-centre corridor is marked
    drivable. Everything else below the horizon is conservatively treated as
    non-traversable by the mapper.
    """

    ROAD_OBJECTS = {
        "person": "person",
        "bicycle": "bicycle",
        "car": "vehicle",
        "motorcycle": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "cat": "animal",
        "dog": "animal",
        "backpack": "obstacle",
        "handbag": "obstacle",
        "suitcase": "obstacle",
    }

    def __init__(
        self,
        config: Dict[str, object],
        horizon_fraction: float = 0.25,
        camera_config: Dict[str, object] | None = None,
    ):
        camera = camera_config or {}
        self.width = int(camera.get("width", 640))
        self.height = int(camera.get("height", 480))
        self.horizon_fraction = float(horizon_fraction)
        self.min_area = int(config.get("minimum_obstacle_area_px", 160))
        self.min_width = int(config.get("minimum_obstacle_width_px", 8))
        self.min_height = int(config.get("minimum_obstacle_height_px", 8))
        self.dark_ditch_value = int(config.get("dark_ditch_value", 52))
        self.tree_min_area = int(config.get("tree_min_area_px", 900))
        self.ema_alpha = float(config.get("ground_adaptation_alpha", 0.15))
        self.camera_height = float(camera.get("camera_height_m", 0.0))
        self.max_range = float(camera.get("max_ground_range_m", 12.0))
        self.focal = float(camera.get("focal_length_px", 0.0))
        self.half_fov_tan = np.tan(np.deg2rad(float(camera.get("horizontal_fov_degrees", 76.0)) / 2.0))

        self.ego_mean_s = 20.0
        self.ego_mean_v = 100.0
        self.ego_mean_h = 20.0
        self.ego_initialized = False
        self.yolo = None
        self.yolo_confidence = 0.38
        self.yolo_interval = 1
        self._frame_number = 0
        self.object_detector_status = "disabled"
        self._load_yolo(config.get("object_detector", {}))
        self.yolop = YOLOPAdapter(config.get("yolop", {}) if isinstance(config.get("yolop", {}), dict) else {})
        self.yolop_status = self.yolop.status
        self._last_vegetation = np.zeros((self.height, self.width), dtype=bool)
        self._last_lane = np.zeros((self.height, self.width), dtype=bool)

    def _load_yolo(self, detector_config: object) -> None:
        settings = detector_config if isinstance(detector_config, dict) else {}
        if not bool(settings.get("enabled", False)):
            return
        self.yolo_confidence = float(settings.get("confidence", 0.38))
        self.yolo_interval = max(1, int(settings.get("inference_interval_frames", 1)))
        try:
            from ultralytics import YOLO

            self.yolo = YOLO(str(settings.get("model_path", "yolov8n.pt")))
            self.object_detector_status = "yolo_enabled"
        except (ImportError, OSError, RuntimeError, ValueError):
            # The core safe-ground path remains usable without the optional
            # neural detector, which is important for an offline field setup.
            self.object_detector_status = "yolo_unavailable"

    def detect(self, frame_bgr: np.ndarray) -> PerceptionResult:
        """Return autonomy-compatible drivable and obstacle evidence."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("TerrainPerception expects a BGR colour frame")
        h, w = frame_bgr.shape[:2]
        horizon = min(max(0, int(h * self.horizon_fraction)), h - 1)
        classes, vegetation = self._segment_safe_ground(frame_bgr, horizon)
        yolop_masks = self.yolop.segment(frame_bgr)
        self.yolop_status = self.yolop.status
        if yolop_masks is not None:
            yolop_drivable, self._last_lane = yolop_masks
            # A model may be trained on a road domain different from the UGV's
            # terrain. Intersecting it with the safe-ground corridor can only
            # remove unsafe space; it can never invent a drivable shortcut.
            classes[(classes == TRAVERSABLE_PATH) & ~yolop_drivable] = ROUGH_TERRAIN
        else:
            self._last_lane = np.zeros((h, w), dtype=bool)
        self._last_vegetation = vegetation
        drivable = np.where(classes == TRAVERSABLE_PATH, 255, 0).astype(np.uint8)
        # Rough terrain is intentionally occupied: it has not been verified as
        # safe and must not become a shortcut in the local planner.
        rows = np.indices((h, w))[0]
        obstacle = np.where((classes != TRAVERSABLE_PATH) & (rows >= horizon), 255, 0).astype(np.uint8)
        hazards = self._semantic_hazards(classes, vegetation, horizon)
        self._merge_yolo_hazards(frame_bgr, obstacle, hazards, horizon)
        return PerceptionResult(drivable, obstacle, self._annotate(frame_bgr, classes, hazards, self._last_lane), hazards)

    def _segment_safe_ground(self, frame_bgr: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Port the safe-ground policy without adding a depth-camera dependency."""
        h, w = frame_bgr.shape[:2]
        classes = np.full((h, w), BACKGROUND, dtype=np.uint8)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = (hsv[:, :, index] for index in range(3))
        blue = frame_bgr[:, :, 0].astype(np.float32)
        green = frame_bgr[:, :, 1].astype(np.float32)
        red = frame_bgr[:, :, 2].astype(np.float32)
        texture = np.std(frame_bgr, axis=2)
        intensity = blue + green + red + 1e-4
        chroma_deviation = (np.abs(red / intensity - 0.3333) + np.abs(green / intensity - 0.3333) + np.abs(blue / intensity - 0.3333))

        valid_rows = np.zeros((h, w), dtype=bool)
        valid_rows[horizon:, :] = True
        seed_x, seed_y = w // 2, min(h - 15, int(h * 0.95))
        y1, y2 = max(horizon, int(h * 0.84)), min(h - 5, int(h * 0.98))
        x1, x2 = max(0, int(w * 0.42)), min(w, int(w * 0.58))
        if y2 > y1 and x2 > x1:
            sample_h = float(np.mean(hue[y1:y2, x1:x2]))
            sample_s = float(np.mean(saturation[y1:y2, x1:x2]))
            sample_v = float(np.mean(value[y1:y2, x1:x2]))
            if self.ego_initialized:
                alpha = self.ema_alpha
                self.ego_mean_h = (1.0 - alpha) * self.ego_mean_h + alpha * sample_h
                self.ego_mean_s = (1.0 - alpha) * self.ego_mean_s + alpha * sample_s
                self.ego_mean_v = (1.0 - alpha) * self.ego_mean_v + alpha * sample_v
            else:
                self.ego_mean_h, self.ego_mean_s, self.ego_mean_v = sample_h, sample_s, sample_v
                self.ego_initialized = True

        is_painted_curb = np.zeros((h, w), dtype=bool)
        if self.ego_mean_s < 45:
            yellow = ((hue >= 14) & (hue <= 38) & (saturation >= 42) & (value >= 55) & (red >= green - 15) & valid_rows)
            green_curb = ((hue >= 38) & (hue <= 85) & (saturation >= 35) & (value >= 35) & valid_rows & ~yellow)
            curb = cv2.morphologyEx((yellow | green_curb).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
            central = np.zeros((h, w), dtype=bool)
            central[int(h * 0.65):, int(w * 0.35):int(w * 0.65)] = True
            is_painted_curb = curb & ~central

        excess_green = 2.0 * green - red - blue
        is_plants = (((excess_green > 5.0) & (green >= red - 5)) | ((hue >= 24) & (hue <= 98) & (saturation >= 24) & (green >= blue)))
        is_plants &= valid_rows & ~is_painted_curb

        # Conservative monocular cue for a ditch/drop-off. A depth source can
        # later strengthen this class without changing the downstream API.
        raw_ditch = ((value < self.dark_ditch_value) & (saturation < 95) & valid_rows & ~is_painted_curb)
        is_ditch = cv2.morphologyEx(raw_ditch.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        boundaries = is_painted_curb | is_plants | is_ditch

        if self.ego_mean_s < 45:
            road_surface = ((chroma_deviation < 0.18) & (saturation < 55) & (value >= 18) & (value <= 250) & (texture < 30) & valid_rows & ~boundaries)
        else:
            hue_difference = np.minimum(np.abs(hue.astype(float) - self.ego_mean_h), 180.0 - np.abs(hue.astype(float) - self.ego_mean_h))
            road_surface = ((hue_difference < 18) & (np.abs(saturation.astype(float) - self.ego_mean_s) < 48) & (np.abs(value.astype(float) - self.ego_mean_v) < 85) & valid_rows & ~boundaries)

        corridor = road_surface.copy()
        for row in range(h - 1, horizon, -1):
            columns = np.flatnonzero(corridor[row])
            if len(columns) >= 2:
                corridor[row, columns[0]:columns[-1] + 1] = True
            elif row < h - 1:
                previous = np.flatnonzero(corridor[row + 1])
                if len(previous) >= 2:
                    left, right = min(seed_x, previous[0] + 1), max(seed_x, previous[-1] - 1)
                    if left < right:
                        corridor[row, left:right + 1] = True
        corridor[boundaries] = False

        count, labels, stats, _ = cv2.connectedComponentsWithStats(corridor.astype(np.uint8), connectivity=8)
        road_mask = np.zeros((h, w), dtype=bool)
        road_label = labels[seed_y, seed_x]
        if road_label:
            road_mask = labels == road_label
        elif count > 1:
            road_mask = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        road_mask[boundaries] = False

        classes[road_mask] = TRAVERSABLE_PATH
        classes[is_painted_curb] = POSITIVE_OBSTACLE
        classes[is_ditch & ~road_mask] = NEGATIVE_OBSTACLE
        classes[is_plants & ~is_ditch] = ROUGH_TERRAIN
        classes[valid_rows & ~road_mask & (classes == BACKGROUND)] = ROUGH_TERRAIN
        return classes, is_plants

    def _semantic_hazards(self, classes: np.ndarray, vegetation: np.ndarray, horizon: int) -> list[Hazard]:
        hazards: list[Hazard] = []
        for class_id, label, confidence in ((POSITIVE_OBSTACLE, "curb", 0.82), (NEGATIVE_OBSTACLE, "ditch", 0.82)):
            count, _, stats, _ = cv2.connectedComponentsWithStats((classes == class_id).astype(np.uint8), connectivity=8)
            for index in range(1, count):
                x, y, width, height, area = (int(number) for number in stats[index])
                if area < self.min_area or width < self.min_width or height < self.min_height:
                    continue
                if y + height <= horizon or width * height > classes.size * 0.62:
                    continue
                point = self._ground_point(x + width / 2.0, y + height - 2.0, classes.shape)
                hazards.append(Hazard(label, confidence, (x, y, width, height), point))
        # Plants and trees are separately reported for situational awareness.
        # They stay non-traversable regardless of their label, so a false tree
        # label cannot make the path planner less conservative.
        count, _, stats, _ = cv2.connectedComponentsWithStats(vegetation.astype(np.uint8), connectivity=8)
        for index in range(1, count):
            x, y, width, height, area = (int(number) for number in stats[index])
            if area < self.min_area or width < self.min_width or height < self.min_height:
                continue
            if y + height <= horizon or width * height > classes.size * 0.62:
                continue
            aspect = height / max(1, width)
            # A tree crown is typically a substantial component above the
            # near-ground plant band; compact lower components remain plants.
            label = "tree" if area >= self.tree_min_area and y < classes.shape[0] * 0.78 and aspect >= 0.55 else "plant"
            point = self._ground_point(x + width / 2.0, y + height - 2.0, classes.shape)
            hazards.append(Hazard(label, 0.72, (x, y, width, height), point))
        return hazards

    def _merge_yolo_hazards(self, frame: np.ndarray, obstacle: np.ndarray, hazards: list[Hazard], horizon: int) -> None:
        self._frame_number += 1
        if self.yolo is None or self._frame_number % self.yolo_interval:
            return
        try:
            result = self.yolo(frame, verbose=False, conf=self.yolo_confidence)[0]
        except (RuntimeError, ValueError, cv2.error):
            return
        names = result.names
        for box in result.boxes or []:
            label = self.ROAD_OBJECTS.get(str(names[int(box.cls[0].item())]).lower())
            if label is None:
                continue
            x1, y1, x2, y2 = (int(round(number)) for number in box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1 or y2 < horizon:
                continue
            candidate = Hazard(label, float(box.conf[0].item()), (x1, y1, x2 - x1, y2 - y1), self._ground_point((x1 + x2) / 2.0, y2 - 2.0, frame.shape[:2]))
            duplicate = next((index for index, hazard in enumerate(hazards) if _iou(candidate.bbox, hazard.bbox) > 0.35), None)
            if duplicate is None:
                hazards.append(candidate)
            else:
                hazards[duplicate] = candidate
            cv2.rectangle(obstacle, (x1, y1), (x2, y2), 255, -1)

    def _ground_point(self, u: float, v: float, shape: tuple[int, int]) -> tuple[float, float] | None:
        """Return (lateral_m, forward_m), matching the occupancy-map convention."""
        height, width = shape
        horizon = height * self.horizon_fraction
        if v <= horizon:
            return None
        focal = self.focal or width / (2.0 * self.half_fov_tan)
        if self.camera_height > 0.0:
            forward = self.camera_height * focal / max(1.0, v - horizon)
        else:
            forward = self.max_range * (height - v) / max(1.0, height - horizon)
        forward = float(np.clip(forward, 0.35, self.max_range))
        return float((u - width / 2.0) * forward / focal), forward

    @staticmethod
    def _annotate(frame: np.ndarray, classes: np.ndarray, hazards: Iterable[Hazard], lane_mask: np.ndarray) -> np.ndarray:
        output, overlay = frame.copy(), frame.copy()
        for class_id, colour in CLASS_COLOURS_BGR.items():
            overlay[classes == class_id] = colour
        output = cv2.addWeighted(output, 0.68, overlay, 0.32, 0)
        if lane_mask.shape == classes.shape and np.any(lane_mask):
            output[lane_mask] = (0.45 * output[lane_mask] + 0.55 * np.array((255, 255, 0))).astype(np.uint8)
        for hazard in hazards:
            x, y, width, height = hazard.bbox
            if hazard.label == "ditch":
                colour = CLASS_COLOURS_BGR[NEGATIVE_OBSTACLE]
            elif hazard.label in {"tree", "plant"}:
                colour = CLASS_COLOURS_BGR[ROUGH_TERRAIN]
            else:
                colour = CLASS_COLOURS_BGR[POSITIVE_OBSTACLE]
            cv2.rectangle(output, (x, y), (x + width, y + height), colour, 2)
            distance = f" {hazard.ground_point[1]:.1f}m" if hazard.ground_point else ""
            cv2.putText(output, f"{hazard.label}{distance} {hazard.confidence:.0%}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        return output


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1, x2, y2 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, aw * ah + bw * bh - intersection)
