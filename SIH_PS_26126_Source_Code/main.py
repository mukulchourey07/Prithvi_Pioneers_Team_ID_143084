"""Run the GPS-free UGV autonomy prototype from a simulator, webcam, or video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugv_autonomy.runtime import AutonomyRuntime  # noqa: E402
from ugv_autonomy.simulator import UGVSimulator  # noqa: E402
from ugv_autonomy.control import JsonSerialVehicleAdapter, VehicleAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera-only UGV perception, visual odometry, planning and control.")
    parser.add_argument("--source", choices=("sim", "webcam", "video"), default="sim", help="Input image source.")
    parser.add_argument("--video", help="Path to front-camera video (required with --source video).")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for --source webcam.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json", help="JSON configuration path.")
    parser.add_argument("--display", action="store_true", help="Show annotated camera and local map windows.")
    parser.add_argument("--record-dir", type=Path, help="Optional directory for annotated-camera and map MP4 files.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames; simulation defaults to 280.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_vehicle(config: dict) -> VehicleAdapter:
    vehicle_config = config.get("vehicle", {})
    if isinstance(vehicle_config, dict) and vehicle_config.get("type") == "json_serial":
        return JsonSerialVehicleAdapter(vehicle_config)
    return VehicleAdapter()


def main() -> int:
    args = parse_args()
    if args.source == "video" and not args.video:
        raise SystemExit("--video is required when --source video is selected")
    config = load_config(args.config)
    runtime = AutonomyRuntime(config, vehicle=build_vehicle(config))
    simulator = UGVSimulator(config["camera"]) if args.source == "sim" else None
    capture = None
    if simulator is None:
        capture = cv2.VideoCapture(args.camera_index if args.source == "webcam" else args.video)
        if not capture.isOpened():
            raise SystemExit("Could not open camera/video input")

    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=True)
    camera_writer = map_writer = None
    max_frames = args.max_frames if args.max_frames is not None else (280 if simulator else 0)
    frame_index = 0
    try:
        while not max_frames or frame_index < max_frames:
            if simulator:
                frame = simulator.frame()
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (int(config["camera"]["width"]), int(config["camera"]["height"])))
            output = runtime.process(frame)
            if simulator:
                simulator.apply(output.command)
            if args.record_dir:
                if camera_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = output.perception.annotated.shape[:2]
                    mh, mw = output.map_image.shape[:2]
                    camera_writer = cv2.VideoWriter(str(args.record_dir / "annotated_camera.mp4"), fourcc, 10.0, (w, h))
                    map_writer = cv2.VideoWriter(str(args.record_dir / "local_map.mp4"), fourcc, 10.0, (mw, mh))
                camera_writer.write(output.perception.annotated)
                map_writer.write(output.map_image)
            if args.display:
                cv2.imshow("UGV autonomy: camera perception", output.perception.annotated)
                cv2.imshow("UGV autonomy: local map and replan", output.map_image)
                cv2.imshow("UGV autonomy: localisation trajectory", output.localisation_image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if frame_index % 25 == 0:
                pose = output.motion.pose
                print(f"frame={frame_index:03d} pose=({pose.x:+.2f}, {pose.y:+.2f}, {pose.yaw:+.2f}) " f"vo={output.motion.method} route={output.plan.status} cmd={output.command.reason}")
            if output.command.reason == "destination_reached":
                print("Destination reached safely.")
                break
            if simulator and simulator.collided:
                print("Simulator collision: command stopped. Inspect perception/map calibration.")
                return 2
            frame_index += 1
    finally:
        if capture is not None:
            capture.release()
        if camera_writer is not None:
            camera_writer.release()
            map_writer.release()
        runtime.vehicle.close()
        cv2.destroyAllWindows()
    print(f"Finished after {frame_index} frames. Local pose is visual-odometry only; no GPS was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
