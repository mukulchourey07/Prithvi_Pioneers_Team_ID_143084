"""Optional official YOLOP panoptic-driving inference adapter.

YOLOP jointly emits object detections, a drivable-area mask, and a lane mask.
The adapter is deliberately optional: a missing PyTorch/YOLOP installation must
never prevent the conservative safe-ground detector from stopping the UGV.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class YOLOPAdapter:
    """Load the official HUST YOLOP Hub model only when explicitly enabled."""

    def __init__(self, config: Dict[str, object]):
        self.enabled = bool(config.get("enabled", False))
        self.input_size = int(config.get("input_size", 640))
        self.min_drivable_coverage = float(config.get("min_drivable_coverage", 0.03))
        self.status = "disabled"
        self.model = None
        self.torch = None
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            import torch

            # This is the official PyTorch Hub entry point for YOLOP. Its
            # pretrained weights are cached by Torch after the first download.
            self.model = torch.hub.load("hustvl/yolop", "yolop", pretrained=True, trust_repo=True)
            self.model.eval()
            self.torch = torch
            self.status = "yolop_enabled"
        except (ImportError, OSError, RuntimeError, ValueError):
            self.model = None
            self.status = "yolop_unavailable"

    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return boolean drivable-area and lane masks at the frame resolution."""
        if self.model is None or self.torch is None:
            return None
        height, width = frame_bgr.shape[:2]
        image = cv2.resize(frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        try:
            with self.torch.no_grad():
                _, drivable_logits, lane_logits = self.model(tensor)
            drivable = self.torch.argmax(drivable_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8) > 0
            lane = self.torch.argmax(lane_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8) > 0
        except (RuntimeError, ValueError, TypeError):
            self.status = "yolop_inference_failed"
            return None
        drivable = cv2.resize(drivable.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        lane = cv2.resize(lane.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        if float(np.mean(drivable)) < self.min_drivable_coverage:
            # Reject an obviously invalid segmentation result rather than
            # constraining the planner with a blank model output.
            return None
        return drivable, lane
