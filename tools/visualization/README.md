# Visualization Tools

This directory contains two visualization utilities for FFPS on SemanticKITTI.

## 1. View SemanticKITTI GT labels

Use `view_semkitti_gt_label.py` to open the official SemanticKITTI voxel viewer for `.label` GT data.

Example:

```bash
python tools/visualization/view_semkitti_gt_label.py \
  --dataset data/SemanticKITTI/dataset \
  --sequence 08 \
  --frame 000645
```

## 2. View FFPS prediction `.pkl` files

Use `visualize_semkitti_pred_pkl_mayavi.py` to render FFPS voxel predictions saved in `.pkl` format.

Example:

```bash
python tools/visualization/visualize_semkitti_pred_pkl_mayavi.py \
  path/to/000645.pkl \
  --dataset semantic-kitti \
  --calib data/SemanticKITTI/dataset/sequences/08/calib.txt \
  --show
```

## Visualization Environment

The visualization tools are best installed in a separate environment from training.

```bash
conda create -n ffps-vis python=3.9 -y
conda activate ffps-vis
pip install -r tools/visualization/requirements-visualization.txt
```
