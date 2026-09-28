# Vision-Guided UGV Autonomy

An end-to-end, camera-first autonomy prototype for the Smart India Hackathon use case:

```text
Camera -> safe-ground / hazard perception -> visual odometry -> safe route planner -> drive commands
```

It runs without GPS. The default deterministic simulation demonstrates adaptive safe-ground segmentation, curb/vegetation/ditch exclusion, YOLO road-object detection, visual motion estimation from image features, dynamic occupancy mapping, A* replanning, and steering/throttle commands. A webcam or recorded video can be used in place of the simulator.

## Quick start

Use Python 3.10+.

```powershell
cd ugv_autonomy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python main.py --source sim --display
```

If your OpenCV package is a minimal build without `HOGDescriptor`, the program
automatically disables that optional people-detector extension. The core terrain
and hazard pipeline still runs. To enable it where supported, install the full
package: `pip uninstall opencv opencv-python-headless -y; pip install opencv-python`.

Close the display window or press `q` to stop. The simulation saves a short camera/map recording when `--record-dir recordings` is supplied.

## Camera calibration

1. Print or display a flat checkerboard, for example **9 x 6 inner corners** with 25 mm squares.
2. Mount the actual UGV camera. Capture 20--30 sharp images while the board is tilted, moved to every part of the image, and held at several distances. All images must have the same resolution used at runtime.
3. Run the calibration utility:

```powershell
python calibrate_camera.py --images calibration_images --pattern-cols 9 --pattern-rows 6 --square-mm 25 --show
```

The script writes the focal length, principal point, horizontal FOV, and lens-distortion coefficients into `config.json`. Runtime automatically undistorts each incoming frame before perception and visual odometry.

4. With the camera fixed in its final vehicle mount and the UGV on level ground, measure the vertical pixel of the far flat-ground horizon/vanishing line. Set `camera.horizon_fraction` to `horizon_pixel / image_height` in `config.json`.
5. Put cones at known distances (2 m, 5 m, 8 m, 12 m) and tune `max_ground_range_m` until their occupancy-grid positions match. Finally, drive a straight measured track at a consistent frame rate/speed and tune `vo_scale_m_per_frame` so the visual-odometry trajectory matches the measured distance.

Keep mean reprojection error below roughly 0.5 pixels. Monocular scale varies with speed and frame rate; a stereo/depth camera or visual-inertial setup is recommended for reliable metric position.

## Other sources

```powershell
# Default Windows webcam
python main.py --source webcam --camera-index 0 --display

# A recorded camera video
python main.py --source video --video path\to\front_camera.mp4 --display
```

## Monocular SLAM with one camera

The standard runtime uses visual odometry. This project can instead use the native ORB-SLAM3 monocular engine for keyframes, persistent landmarks, relocalisation, and loop closure. ORB-SLAM3 needs a Linux-supported Python binding, its `ORBvoc.txt` vocabulary file, and a camera settings YAML; it is not bundled in this ZIP. If any required item is missing, runtime now safely falls back to visual odometry instead of crashing.

1. Calibrate the camera first and **measure the vertical distance from the ground to the centre of the camera lens in metres**. Enter it in `config.json`, for example `"camera_height_m": 0.37`. This gives the occupancy grid metric ground distances through calibrated flat-ground projection.
2. Install ORB-SLAM3 on Ubuntu (or a supported Linux computer) and obtain `ORBvoc.txt`. The binding package is installed with:

```bash
pip install -r requirements-slam.txt
```

3. Generate a matching ORB-SLAM3 camera YAML after calibration:

```bash
python create_orbslam_settings.py --fps 20
```

4. Copy/link `ORBvoc.txt` to `slam/ORBvoc.txt`, set `slam.enabled` to `true`, and start the normal webcam/video command. The runtime selects ORB-SLAM3 automatically; it retains visual odometry when SLAM is disabled.

```json
"slam": {
  "enabled": true,
  "vocabulary_path": "slam/ORBvoc.txt",
  "settings_path": "slam/ugv_camera.yaml",
  "metric_scale_m_per_slam_unit": 1.0,
  "pose_convention": "Tcw"
}
```

Monocular SLAM has an unknown translation scale. Camera height improves the metric obstacle projection but does not by itself provide SLAM scale. Drive a measured, straight 5 m route and adjust `metric_scale_m_per_slam_unit` until the reported path length is 5 m. Keep the camera rigid and level, move it smoothly during initialisation, and always test with motors disconnected first.

When `--display` is used, the **localisation trajectory** window explicitly says either `ORB-SLAM3 MONOCULAR` or `VISUAL ODOMETRY FALLBACK`. If it says fallback, SLAM is not running: check that `slam.enabled` is true and that the vocabulary, YAML settings, and native ORB-SLAM3 binding are installed.

With real imagery, set the camera parameters in `config.json` after doing an on-site calibration. The safe-ground detector adapts its local ground colour model from the patch directly in front of the UGV and is deliberately conservative: only the bottom-centre connected corridor is marked drivable, while unrecognised terrain becomes obstacle evidence for the map. Tune `perception.dark_ditch_value` on site for the camera exposure and surface conditions. For production, validate against representative terrain and, if available, feed calibrated depth into a semantic-segmentation model using the same `PerceptionResult` interface.

## Better person and vehicle detection

The safe-ground integration includes the supplied YOLOv8n model (`yolov8n.pt`) and enables it by default for **person, car, truck, bus, motorcycle, bicycle, animal, and carried-object** detection. Install the dependencies with:

```powershell
pip install -r requirements-objects.txt
```

Set `perception.object_detector.enabled` to `false` for a CPU-only or offline safe-ground-only run. Tune `confidence` upward (for example, `0.55`) if detections are still noisy, and tune `inference_interval_frames` to meet the camera frame-rate budget. Do not use a real vehicle until detections are field-tested with an independent emergency stop.

## YOLOP driving perception, trees, and plants

The project now has an optional adapter for the official HUST **YOLOP** model. It jointly produces drivable-area and lane masks; the runtime intersects its drivable output with the already conservative safe-ground corridor, so model output can never create an unverified shortcut. Enable it only after installing its PyTorch dependencies and allowing the first model download:

```powershell
pip install -r requirements-yolop.txt
```

Then set `perception.yolop.enabled` to `true` in `config.json`. The first enabled run loads the official PyTorch Hub model. There is no official model/checkpoint called “YOLOPv3” in the upstream HUST YOLOP release; this integration uses that official YOLOP interface rather than silently substituting an unrelated model. For a trained third-generation model, supply its repository, checkpoint, input/output definition, and licence before replacing the adapter.

Safe-ground segmentation also reports vegetation components as `plant` or `tree` and keeps them non-traversable. These are classical visual labels, so validate their thresholding against the camera's lighting and vegetation before relying on them operationally.

## Verified-space overtaking

For a person, vehicle, bicycle, animal, or carried object ahead in the UGV's central path, the stack evaluates left and right passing waypoints. It starts an overtake only when the local map has already observed a robot-width-safe drivable corridor to that waypoint. It then sends the existing pure-pursuit controller a reduced-speed path. Unknown ground, curbs, ditches, plants, and trees are rejected as passing space. If neither side is verified, the UGV stops/replans rather than attempting a pass.

Tune the `overtake` block in `config.json` after measured field tests. This is a low-speed UGV side-pass feature, not a road-traffic passing system; it must not be used near moving traffic, pedestrians, or public roads without redundant sensing and an independent safety controller.

## Motor and steering interface

Pure pursuit emits normalized `steering` (-1 to +1) and `throttle` (0 to 1) values through `VehicleAdapter`. A disarmed `json_serial` option is included for UGV controllers that accept newline-delimited UTF-8 JSON:

```json
{"steering": 0.12, "throttle": 0.28, "emergency_stop": false}
```

Install `requirements-hardware.txt`, enter the correct serial port, and leave `vehicle.armed` set to `false` until bench testing is complete. Only then change `vehicle.type` to `json_serial` and set `armed` to `true`. An unknown serial/CAN protocol must be implemented as its own `VehicleAdapter`; never guess a motor-controller protocol.

## Safety and engineering notes

- This is a prototype and **must not command a physical vehicle without a tested safety controller, hardware E-stop, speed limits, and supervised field validation**.
- Monocular visual odometry has an inherent scale ambiguity. `vo_scale_m_per_frame` is a calibration prior. A stereo/depth camera or camera+IMU improves metric accuracy and relocalisation.
- No GPS API, telemetry, or GPS coordinates are used. In simulation, the hidden ground truth exists only to render the virtual camera and evaluate the estimator.
- The supplied motor command is an abstract `{steering, throttle, emergency_stop}` message. Implement a `VehicleAdapter` with protocol checks for your UGV motor controller.

## Project layout

```text
main.py                         CLI entry point
src/ugv_autonomy/perception.py  Ground and hazard masks / labels
src/ugv_autonomy/visual_odometry.py  ORB + optical-flow visual odometry
src/ugv_autonomy/mapping.py     Image-ground projection and occupancy grid
src/ugv_autonomy/planning.py    Inflated-grid A* replanning
src/ugv_autonomy/control.py     Pure-pursuit steering and safe motor messages
src/ugv_autonomy/runtime.py     End-to-end pipeline
src/ugv_autonomy/simulator.py   GPS-free demo camera world
```

## Flow-chart mapping

| Flow-chart block | Implementation |
|---|---|
| On-board camera | `FrameSource`, OpenCV webcam/video/simulator |
| See safe ground & obstacles | `TerrainPerception` (adaptive safe-ground corridor + YOLO hazards) |
| Estimate position / direction without GPS | `VisualOdometry` |
| Plan a safe route | `OccupancyGrid` + `AStarPlanner` |
| Send steering / motor commands | `PurePursuitController` + `VehicleAdapter` |
