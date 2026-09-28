"""Generate a camera-calibration YAML file required by ORB-SLAM3 monocular mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ORB-SLAM3 camera settings from this project's config.json.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "slam" / "ugv_camera.yaml")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera, vo = config["camera"], config["vo"]
    coefficients = list(camera.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])) + [0.0] * 5
    output = f'''%YAML:1.0
Camera.type: "PinHole"
Camera.fx: {float(vo["focal_length_px"]):.8f}
Camera.fy: {float(vo["focal_length_px"]):.8f}
Camera.cx: {float(vo["principal_point_x"]):.8f}
Camera.cy: {float(vo["principal_point_y"]):.8f}
Camera.k1: {float(coefficients[0]):.10f}
Camera.k2: {float(coefficients[1]):.10f}
Camera.p1: {float(coefficients[2]):.10f}
Camera.p2: {float(coefficients[3]):.10f}
Camera.k3: {float(coefficients[4]):.10f}
Camera.width: {int(camera["width"])}
Camera.height: {int(camera["height"])}
Camera.fps: {args.fps:.2f}
Camera.RGB: 0
ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
