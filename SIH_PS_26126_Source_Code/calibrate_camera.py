"""Calibrate the UGV camera from checkerboard photographs and update config.json.

Example:
    python calibrate_camera.py --images calibration_images --pattern-cols 9 --pattern-rows 6 --square-mm 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV checkerboard camera calibration for the UGV.")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing 15--30 checkerboard images.")
    parser.add_argument("--pattern-cols", type=int, required=True, help="Number of INNER checkerboard corners across.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Number of INNER checkerboard corners down.")
    parser.add_argument("--square-mm", type=float, required=True, help="Physical edge length of one square in millimetres.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="Configuration JSON to update.")
    parser.add_argument("--show", action="store_true", help="Show detected corners while processing images.")
    return parser.parse_args()


def image_paths(folder: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in supported)


def reprojection_error(object_points, image_points, rotations, translations, matrix, distortion) -> float:
    total_error, total_points = 0.0, 0
    for object_set, image_set, rotation, translation in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(object_set, rotation, translation, matrix, distortion)
        total_error += cv2.norm(image_set, projected, cv2.NORM_L2) ** 2
        total_points += len(object_set)
    return float(np.sqrt(total_error / total_points))


def main() -> int:
    args = parse_args()
    paths = image_paths(args.images)
    if len(paths) < 12:
        raise SystemExit("Use at least 12 images; 20--30 varied checkerboard views are recommended.")
    pattern = (args.pattern_cols, args.pattern_rows)
    object_template = np.zeros((args.pattern_cols * args.pattern_rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
    object_template *= args.square_mm / 1000.0  # Store physical dimensions in metres.
    object_points, image_points = [], []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable file: {path.name}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        if image_size != (gray.shape[1], gray.shape[0]):
            print(f"Skipping {path.name}: resolution differs from the first image")
            continue
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        else:
            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not found:
            print(f"No checkerboard found: {path.name}")
            continue
        object_points.append(object_template)
        image_points.append(corners)
        print(f"Accepted: {path.name}")
        if args.show:
            cv2.drawChessboardCorners(image, pattern, corners, found)
            cv2.imshow("Checkerboard corners (press q to stop)", image)
            if cv2.waitKey(120) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    if len(object_points) < 12 or image_size is None:
        raise SystemExit(f"Only {len(object_points)} valid images. Capture more varied, sharp checkerboard images.")

    _, matrix, distortion, rotations, translations = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    error = reprojection_error(object_points, image_points, rotations, translations, matrix, distortion)
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    fov_x = float(np.degrees(2 * np.arctan(image_size[0] / (2 * fx))))
    config["camera"].update({
        "width": image_size[0],
        "height": image_size[1],
        "horizontal_fov_degrees": round(fov_x, 4),
        "distortion_coefficients": [round(float(value), 9) for value in distortion.ravel()],
    })
    config["vo"].update({
        "focal_length_px": round(float((fx + fy) / 2), 4),
        "principal_point_x": round(float(cx), 4),
        "principal_point_y": round(float(cy), 4),
    })
    with args.config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print("\nCalibration complete")
    print(f"Accepted images: {len(object_points)}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Horizontal FOV: {fov_x:.2f} degrees")
    print(f"Mean reprojection error: {error:.3f} pixels (under 0.5 px is a good target)")
    print(f"Updated: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
