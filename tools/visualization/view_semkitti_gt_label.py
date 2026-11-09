#!/usr/bin/env python3

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_official_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(
        "semantic_kitti_visualize_voxels", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_sequence_directory(base_dir: Path, sequence: str) -> Path:
    direct_voxels = base_dir / "voxels"
    direct_predictions = base_dir / "predictions"
    if direct_voxels.exists() or direct_predictions.exists():
        return base_dir

    sequence_dir = base_dir / "sequences" / sequence
    if sequence_dir.exists():
        return sequence_dir

    raise FileNotFoundError(
        "Could not find a usable voxel directory. Expected either "
        f"'{direct_voxels}' or '{sequence_dir / 'voxels'}'."
    )


def find_frame_index(window, frame_id: str) -> int:
    subdir = window.subdirs[window.current_subdir]
    current_data = window.current_data
    files = window.data[subdir][current_data]
    frame_names = [Path(file_path).stem for file_path in files]

    if frame_id not in frame_names:
        raise ValueError(
            f"Frame {frame_id} not found in {subdir}/{current_data}. "
            f"Available frames: {', '.join(frame_names)}"
        )

    return frame_names.index(frame_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SemanticKITTI voxel viewer for GT voxel labels."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=".",
        help=(
            "Dataset root. Supports either the official layout "
            "'<dataset>/sequences/<sequence>/' or a direct local folder containing "
            "'voxels/' or 'predictions/'. Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--sequence",
        "-s",
        default="00",
        help="Sequence id used when --dataset points to an official SemanticKITTI root.",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="Optional frame id to open first, for example 000645.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.dataset).resolve()
    script_dir = Path(__file__).resolve().parent
    official_module_path = script_dir / "semantic_kitti_official" / "visualize_voxels.py"

    if not official_module_path.exists():
        raise FileNotFoundError(
            f"Official viewer was not found at {official_module_path}"
        )

    if not (script_dir / "auxiliary").exists():
        raise FileNotFoundError(f"Missing auxiliary directory in {script_dir}")

    if not (script_dir / "config" / "semantic-kitti.yaml").exists():
        raise FileNotFoundError(
            f"Missing config file: {script_dir / 'config' / 'semantic-kitti.yaml'}"
        )

    sequence_directory = resolve_sequence_directory(base_dir, args.sequence)
    script_dir_str = str(script_dir)
    if script_dir_str not in sys.path:
        sys.path.insert(0, script_dir_str)
    official = load_official_module(official_module_path)

    old_cwd = Path.cwd()
    os.chdir(script_dir)
    try:
        window = official.Window()
        window.open_directory(str(sequence_directory))

        if args.frame is not None:
            frame_index = find_frame_index(window, args.frame)
            window.currentTimestep = frame_index
            window.sliderValue = frame_index

        window.run()
        official.glfw.terminate()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()

