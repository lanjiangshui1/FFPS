#!/usr/bin/env python3
"""Visualize FFPS voxel predictions stored as pickle files.

This script renders SemanticKITTI or KITTI360 semantic voxel predictions saved
as `.pkl` files. It expects each pickle payload to contain an `output_voxel`
entry and does not require ground-truth labels, poses, or FOV masks.

Examples
--------
Render every `.pkl` file inside a prediction directory and save PNGs:
    python tools/visualization/visualize_semkitti_pred_pkl_mayavi.py path/to/predictions

Render a single frame interactively:
    python tools/visualization/visualize_semkitti_pred_pkl_mayavi.py path/to/000645.pkl --show --no-save

Render a full directory with lighter subsampling:
    python tools/visualization/visualize_semkitti_pred_pkl_mayavi.py path/to/predictions --max-voxels 180000
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


SEMANTIC_KITTI_COLORS = np.array(
    [
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
    ],
    dtype=np.uint8,
)

KITTI360_COLORS = np.concatenate(
    (
        SEMANTIC_KITTI_COLORS[0:6],
        SEMANTIC_KITTI_COLORS[8:15],
        SEMANTIC_KITTI_COLORS[16:],
        np.array([[250, 150, 0, 255], [50, 255, 255, 255]], dtype=np.uint8),
    ),
    axis=0,
)

DEFAULT_GRID_SHAPE = (256, 256, 32)
DEFAULT_VOXEL_ORIGIN = (0.0, -25.6, -2.0)
DEFAULT_CAMERA_IMAGE_SIZES = {
    "semantic-kitti": (1220, 370),
    "kitti360": (1408, 376),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FFPS voxel predictions stored as pickle files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="A directory containing .pkl files or a single .pkl file. Defaults to the current directory.",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="Optional frame id, for example 000645. Only matching files are rendered.",
    )
    parser.add_argument(
        "--dataset",
        choices=("semantic-kitti", "kitti360"),
        default="semantic-kitti",
        help="Color palette preset. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--prediction-key",
        default="output_voxel",
        help="Dictionary key inside each pickle containing the semantic voxel prediction.",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_GRID_SHAPE,
        help="Voxel grid shape used when the stored prediction is flattened. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.2,
        help="Voxel edge length in meters. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        metavar=("OX", "OY", "OZ"),
        default=DEFAULT_VOXEL_ORIGIN,
        help="Voxel grid origin in meters. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Label id to suppress during visualization. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--empty-label",
        type=int,
        default=0,
        help="Label id treated as empty space. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=250000,
        help=(
            "Upper bound for drawn voxels. Larger volumes are uniformly subsampled "
            "to keep Mayavi responsive. Use 0 to disable subsampling."
        ),
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.75,
        help="Cube glyph scale multiplier relative to voxel size. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--save-dir",
        default="tools/visualization/outputs/figures",
        help="Directory used for saved renders. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--calib",
        default="calib.txt",
        help="KITTI-style calibration file used to draw the camera frustum.",
    )
    parser.add_argument(
        "--camera",
        default="P2",
        help="Projection matrix key used for the camera frustum, for example P2 or P3.",
    )
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=7.0,
        help="Depth in meters used to draw the image plane of the camera frustum.",
    )
    parser.add_argument(
        "--frustum-line-width",
        type=float,
        default=2.0,
        help="Line width used to draw the camera frustum.",
    )
    parser.add_argument(
        "--camera-axis-length",
        type=float,
        default=1.5,
        help="Axis length in meters for the camera coordinate triad.",
    )
    parser.add_argument(
        "--no-frustum",
        action="store_true",
        help="Disable drawing the camera frustum even when calib.txt is available.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Mayavi window instead of only saving images.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable PNG export.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(960, 720),
        help="Saved/rendered figure size. Defaults to %(default)s.",
    )
    return parser.parse_args()


def resolve_input_files(path_arg: str, frame: str | None) -> list[Path]:
    input_path = Path(path_arg)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.pkl"))

    if frame is not None:
        files = [path for path in files if path.stem == frame]

    if not files:
        raise FileNotFoundError(
            f"No pickle files matched in {input_path} with frame={frame!r}"
        )

    return files


def load_prediction_volume(
    file_path: Path, prediction_key: str, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    with file_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"{file_path} does not contain a dictionary payload.")

    if prediction_key not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(
            f"{file_path} is missing key {prediction_key!r}. Available keys: {available}"
        )

    volume = np.asarray(payload[prediction_key])
    if volume.ndim == 1:
        expected = int(np.prod(grid_shape))
        if volume.size != expected:
            raise ValueError(
                f"{file_path} contains {volume.size} voxels under {prediction_key!r}, "
                f"but grid-shape {grid_shape} expects {expected}."
            )
        volume = volume.reshape(grid_shape)
    elif volume.ndim != 3:
        raise ValueError(
            f"{file_path} stores {prediction_key!r} with shape {volume.shape}, "
            "but this script expects a 1D flattened grid or a 3D volume."
        )

    return volume.astype(np.uint8, copy=False)


def voxel_centers(
    volume: np.ndarray,
    voxel_size: float,
    origin: tuple[float, float, float],
    empty_label: int,
    ignore_label: int | None,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    visible_mask = volume != empty_label
    if ignore_label is not None:
        visible_mask &= volume != ignore_label

    x_idx, y_idx, z_idx = np.nonzero(visible_mask)
    labels = volume[visible_mask].astype(np.int32, copy=False)

    if labels.size == 0:
        return np.empty((0, 3), dtype=np.float32), labels

    if max_voxels > 0 and labels.size > max_voxels:
        sample_idx = np.linspace(0, labels.size - 1, max_voxels, dtype=np.int64)
        x_idx = x_idx[sample_idx]
        y_idx = y_idx[sample_idx]
        z_idx = z_idx[sample_idx]
        labels = labels[sample_idx]

    coords = np.empty((labels.size, 3), dtype=np.float32)
    coords[:, 0] = origin[0] + (x_idx.astype(np.float32) + 0.5) * voxel_size
    coords[:, 1] = origin[1] + (y_idx.astype(np.float32) + 0.5) * voxel_size
    coords[:, 2] = origin[2] + (z_idx.astype(np.float32) + 0.5) * voxel_size
    return coords, labels


def world_to_plot_coords(points: np.ndarray) -> np.ndarray:
    plot_points = np.empty_like(points, dtype=np.float32)
    # Match the official SemanticKITTI viewer conversion matrix.
    plot_points[:, 0] = -points[:, 1]
    plot_points[:, 1] = points[:, 2]
    plot_points[:, 2] = -points[:, 0]
    return plot_points


def palette_for_dataset(dataset: str) -> np.ndarray:
    if dataset == "semantic-kitti":
        return SEMANTIC_KITTI_COLORS
    if dataset == "kitti360":
        return KITTI360_COLORS
    raise ValueError(f"Unsupported dataset palette: {dataset}")


def parse_kitti_calibration(file_path: Path) -> dict[str, np.ndarray]:
    calibration: dict[str, np.ndarray] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, content = line.split(":", maxsplit=1)
            values = np.fromstring(content.strip(), sep=" ", dtype=np.float64)
            if values.size == 12 and key.startswith("P"):
                calibration[key] = values.reshape(3, 4)
            elif values.size == 12:
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :4] = values.reshape(3, 4)
                calibration[key] = pose
            else:
                raise ValueError(
                    f"Unsupported calibration entry {key!r} with {values.size} values."
                )
    return calibration


def camera_to_velodyne_transform(
    calibration: dict[str, np.ndarray], camera_key: str
) -> tuple[np.ndarray, np.ndarray]:
    if "Tr" not in calibration:
        raise KeyError("Calibration file is missing 'Tr'.")
    if camera_key not in calibration:
        available = ", ".join(sorted(calibration))
        raise KeyError(
            f"Calibration file is missing camera key {camera_key!r}. Available: {available}"
        )

    projection = calibration[camera_key]
    if projection.shape != (3, 4):
        raise ValueError(
            f"Camera entry {camera_key!r} must be a 3x4 projection matrix, got {projection.shape}."
        )

    fx = float(projection[0, 0])
    fy = float(projection[1, 1])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError(f"Projection matrix {camera_key!r} has invalid focal length.")

    # KITTI projection matrices are rectified and encode the camera-center shift
    # relative to cam0 in the last column.
    tx = float(projection[0, 3]) / fx
    ty = float(projection[1, 3]) / fy
    camera_to_cam0 = np.eye(4, dtype=np.float64)
    camera_to_cam0[0, 3] = -tx
    camera_to_cam0[1, 3] = -ty

    cam0_to_velodyne = np.linalg.inv(calibration["Tr"])
    camera_to_velodyne = cam0_to_velodyne @ camera_to_cam0
    return camera_to_velodyne, projection


def build_camera_overlay(
    calibration_file: Path,
    dataset: str,
    camera_key: str,
    frustum_depth: float,
    axis_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = parse_kitti_calibration(calibration_file)
    camera_to_velodyne, projection = camera_to_velodyne_transform(
        calibration, camera_key
    )

    width, height = DEFAULT_CAMERA_IMAGE_SIZES[dataset]
    fx = float(projection[0, 0])
    fy = float(projection[1, 1])
    cx = float(projection[0, 2])
    cy = float(projection[1, 2])

    image_corners = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float64,
    )
    frustum_points_camera = np.ones((5, 4), dtype=np.float64)
    frustum_points_camera[0, :3] = (0.0, 0.0, 0.0)
    frustum_points_camera[1:, 2] = frustum_depth
    frustum_points_camera[1:, 0] = (
        (image_corners[:, 0] - cx) * frustum_depth / fx
    )
    frustum_points_camera[1:, 1] = (
        (image_corners[:, 1] - cy) * frustum_depth / fy
    )
    frustum_points_velodyne = (
        camera_to_velodyne @ frustum_points_camera.T
    ).T[:, :3]

    axis_points_camera = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [axis_length, 0.0, 0.0, 1.0],
            [0.0, axis_length, 0.0, 1.0],
            [0.0, 0.0, axis_length, 1.0],
        ],
        dtype=np.float64,
    )
    axis_points_velodyne = (camera_to_velodyne @ axis_points_camera.T).T[:, :3]

    return (
        world_to_plot_coords(frustum_points_velodyne),
        world_to_plot_coords(axis_points_velodyne),
    )


def render_volume(
    coords: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    scale_factor: float,
    image_size: tuple[int, int],
    save_path: Path | None,
    show: bool,
    title: str,
    frustum_points: np.ndarray | None = None,
    axis_points: np.ndarray | None = None,
    frustum_line_width: float = 2.0,
) -> None:
    from mayavi import mlab

    offscreen = save_path is not None and not show
    if offscreen:
        mlab.options.offscreen = True

    figure = mlab.figure(bgcolor=(1, 1, 1), size=image_size)
    # Disable scene lights so the rendering stays completely flat.
    for light in figure.scene.light_manager.lights:
        light.activate = False

    if coords.shape[0] > 0:
        plot = mlab.points3d(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            labels,
            mode="cube",
            colormap="viridis",
            scale_factor=voxel_size * scale_factor,
            opacity=1.0,
            vmin=1,
            vmax=max(len(colors), 1),
        )
        plot.glyph.scale_mode = "scale_by_vector"
        plot.module_manager.scalar_lut_manager.lut.table = colors
        # Disable lighting entirely so voxel colors stay flat with no reflections.
        plot.actor.property.lighting = False
        plot.actor.actor.property.lighting = False
        plot.actor.property.specular = 0.0
        plot.actor.actor.property.specular = 0.0
        plot.actor.property.diffuse = 0.0
        plot.actor.actor.property.diffuse = 0.0
        plot.actor.property.ambient = 1.0
        plot.actor.actor.property.ambient = 1.0

    if frustum_points is not None:
        frustum_edges = (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
        )
        for start, end in frustum_edges:
            mlab.plot3d(
                frustum_points[[start, end], 0],
                frustum_points[[start, end], 1],
                frustum_points[[start, end], 2],
                color=(0.0, 0.0, 0.0),
                tube_radius=None,
                line_width=frustum_line_width,
            )
        mlab.points3d(
            frustum_points[0:1, 0],
            frustum_points[0:1, 1],
            frustum_points[0:1, 2],
            mode="sphere",
            scale_factor=0.25,
            color=(0.0, 0.0, 0.0),
        )

    if axis_points is not None:
        axis_specs = (
            ((0, 1), (1.0, 0.15, 0.15)),
            ((0, 2), (0.15, 0.7, 0.15)),
            ((0, 3), (0.15, 0.35, 1.0)),
        )
        for (start, end), color in axis_specs:
            mlab.plot3d(
                axis_points[[start, end], 0],
                axis_points[[start, end], 1],
                axis_points[[start, end], 2],
                color=color,
                tube_radius=None,
                line_width=frustum_line_width,
            )

    mlab.outline(color=(0.45, 0.45, 0.45))
    figure.scene.isometric_view()
    figure.scene.camera.zoom(1.35)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mlab.savefig(str(save_path), size=image_size)

    if show:
        mlab.show()
    else:
        mlab.close(figure)


def main() -> None:
    args = parse_args()
    files = resolve_input_files(args.path, args.frame)
    colors = palette_for_dataset(args.dataset)
    grid_shape = tuple(args.grid_shape)
    origin = tuple(args.origin)
    image_size = tuple(args.image_size)
    save_dir = Path(args.save_dir)
    frustum_points = None
    axis_points = None

    if not args.no_frustum:
        calib_path = Path(args.calib)
        if calib_path.exists():
            frustum_points, axis_points = build_camera_overlay(
                calibration_file=calib_path,
                dataset=args.dataset,
                camera_key=args.camera,
                frustum_depth=args.frustum_depth,
                axis_length=args.camera_axis_length,
            )
        else:
            print(f"Calibration file not found, skipping frustum: {calib_path}")

    for file_path in files:
        volume = load_prediction_volume(file_path, args.prediction_key, grid_shape)
        coords, labels = voxel_centers(
            volume=volume,
            voxel_size=args.voxel_size,
            origin=origin,
            empty_label=args.empty_label,
            ignore_label=args.ignore_label,
            max_voxels=args.max_voxels,
        )
        coords = world_to_plot_coords(coords)

        output_path = None
        if not args.no_save:
            input_tag = file_path.parent.name
            output_path = save_dir / f"{input_tag}_{file_path.stem}.png"

        title = f"{file_path.parent.name}/{file_path.stem} | voxels={labels.size}"
        print(f"Rendering {file_path} -> {output_path or '[interactive only]'}")
        render_volume(
            coords=coords,
            labels=labels,
            colors=colors,
            voxel_size=args.voxel_size,
            scale_factor=args.scale_factor,
            image_size=image_size,
            save_path=output_path,
            show=args.show,
            title=title,
            frustum_points=frustum_points,
            axis_points=axis_points,
            frustum_line_width=args.frustum_line_width,
        )


if __name__ == "__main__":
    main()


