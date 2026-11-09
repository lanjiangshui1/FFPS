import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "open3d is required for visualization. Install it with "
        "`pip install open3d` or `pip install open3d-python`."
    ) from exc


SEMANTIC_KITTI_CLASS_NAMES = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


# Class 0 is kept black and is not visualized by default.
SEMANTIC_KITTI_COLORS = np.array(
    [
        [0, 0, 0],
        [100, 150, 245],
        [100, 230, 245],
        [30, 60, 150],
        [80, 30, 180],
        [100, 80, 250],
        [255, 30, 30],
        [255, 40, 200],
        [150, 30, 90],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [175, 0, 75],
        [255, 200, 0],
        [255, 120, 50],
        [0, 175, 0],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a SemanticKITTI validation prediction saved as a .pkl file."
    )
    parser.add_argument("pkl_path", help="Path to a validation prediction .pkl file.")
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=[256, 256, 32],
        metavar=("X", "Y", "Z"),
        help="Voxel grid size used by the model. Default: 256 256 32.",
    )
    parser.add_argument(
        "--pc-range",
        nargs=6,
        type=float,
        default=[0.0, -25.6, -2.0, 51.2, 25.6, 4.4],
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Point-cloud range used by the model.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Randomly subsample occupied voxels if there are more than this many points.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one voxel every N indices along each axis before rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save the Open3D screenshot and raw image preview.",
    )
    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
        help="Visualizer background color.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Open3D rendered point size.",
    )
    return parser.parse_args()


def load_prediction(pkl_path):
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)

    if "output_voxel" not in data:
        raise KeyError(
            f"`output_voxel` was not found in {pkl_path}. Available keys: {sorted(data.keys())}"
        )

    voxel = np.asarray(data["output_voxel"])
    raw_img = data.get("raw_img", None)
    return voxel, raw_img


def reshape_voxel_grid(voxel, occ_size):
    occ_size = tuple(int(v) for v in occ_size)
    total_size = int(np.prod(occ_size))

    if voxel.ndim == 1:
        if voxel.size != total_size:
            raise ValueError(
                f"Flat voxel array has {voxel.size} elements, but occ_size={occ_size} "
                f"expects {total_size}."
            )
        voxel = voxel.reshape(occ_size)
    elif voxel.ndim != 3:
        raise ValueError(
            f"Unsupported voxel shape {voxel.shape}. Expected flat or 3D voxel data."
        )

    return voxel.astype(np.int32, copy=False)


def voxel_grid_to_points(voxel_grid, pc_range, stride=1, max_points=200000):
    occupied = voxel_grid > 0
    indices = np.argwhere(occupied)

    if indices.size == 0:
        raise ValueError("No occupied voxels were found in this prediction.")

    if stride > 1:
        keep = np.all(indices % stride == 0, axis=1)
        indices = indices[keep]

    labels = voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    if max_points > 0 and len(indices) > max_points:
        rng = np.random.default_rng(0)
        selected = rng.choice(len(indices), size=max_points, replace=False)
        indices = indices[selected]
        labels = labels[selected]

    pc_range = np.asarray(pc_range, dtype=np.float32)
    mins = pc_range[:3]
    maxs = pc_range[3:]
    voxel_size = (maxs - mins) / np.asarray(voxel_grid.shape, dtype=np.float32)

    points = mins + (indices.astype(np.float32) + 0.5) * voxel_size
    colors = SEMANTIC_KITTI_COLORS[labels.clip(0, len(SEMANTIC_KITTI_COLORS) - 1)] / 255.0
    return points, colors, labels


def normalize_raw_image(raw_img):
    if raw_img is None:
        return None

    array = np.asarray(raw_img)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    return array


def save_raw_image(raw_img, save_path):
    array = normalize_raw_image(raw_img)
    if array is None:
        return False

    Image.fromarray(array).save(save_path)
    return True


def print_label_stats(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Visible semantic classes in the rendered voxels:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        name = SEMANTIC_KITTI_CLASS_NAMES[label] if 0 <= label < len(SEMANTIC_KITTI_CLASS_NAMES) else str(label)
        print(f"  {label:>2}: {name:<15} {count}")


def visualize(points, colors, point_size, background, screenshot_path=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SemanticKITTI Prediction Viewer", width=1600, height=900)
    vis.add_geometry(point_cloud)
    vis.add_geometry(coord_frame)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array(
        [0.0, 0.0, 0.0] if background == "black" else [1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    view_control = vis.get_view_control()
    bbox = point_cloud.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = max(float(np.max(bbox.get_extent())), 1.0)
    view_control.set_lookat(center)
    view_control.set_front([0.35, -0.85, -0.4])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(max(0.02, min(0.6, 18.0 / extent)))

    print("Open3D window is ready. Close the window to finish.")
    vis.run()

    if screenshot_path is not None:
        vis.capture_screen_image(str(screenshot_path))
        print(f"Saved screenshot to {screenshot_path}")

    vis.destroy_window()


def main():
    args = parse_args()

    pkl_path = Path(args.pkl_path)
    voxel, raw_img = load_prediction(pkl_path)
    voxel_grid = reshape_voxel_grid(voxel, args.occ_size)
    points, colors, labels = voxel_grid_to_points(
        voxel_grid,
        pc_range=args.pc_range,
        stride=max(1, int(args.stride)),
        max_points=max(0, int(args.max_points)),
    )

    print(f"Loaded: {pkl_path}")
    print(f"Voxel shape: {tuple(voxel_grid.shape)}")
    print(f"Rendered occupied voxels: {len(points)}")
    print_label_stats(labels)

    screenshot_path = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / f"{pkl_path.stem}_open3d.png"
        raw_img_path = output_dir / f"{pkl_path.stem}_raw.png"
        if save_raw_image(raw_img, raw_img_path):
            print(f"Saved raw image preview to {raw_img_path}")
        else:
            print("No raw image was found in this .pkl file.")

    visualize(
        points=points,
        colors=colors,
        point_size=args.point_size,
        background=args.background,
        screenshot_path=screenshot_path,
    )


if __name__ == "__main__":
    main()
