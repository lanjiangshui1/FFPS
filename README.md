# FFPS

FFPS is a vision-centric 3D semantic scene completion model with flow-guided temporal alignment and progressive temporal fusion.

This repository currently supports training and evaluation on SemanticKITTI and SSCBench-KITTI360.

## Model Zoo

We provide the pretrained weights on SemanticKITTI and SSCBench-KITTI360 datasets.

| Dataset | Backbone | Val IoU / mIoU | Test IoU / mIoU | Model Weights |
| --- | --- | --- | --- | --- |
| SemanticKITTI | EfficientNetB7 | 47.9 / 19.2 | 47.1 / 19.8 | [Baidu Pan](https://pan.baidu.com/s/1hCGu-WDYVssGiFZRPeIz-g?pwd=v76m) (code: `v76m`) / [Google Drive](https://drive.google.com/file/d/16MxZmIcSO-2KqByMbXMtkJ71IqozcBls/view?usp=drive_link) |
| SSCBench-KITTI360 | EfficientNetB7 | 49.1 / 22.4 | 48.5 / 21.5 | [Baidu Pan](https://pan.baidu.com/s/1RnL7xJRquqp34Ya5T_6fNQ?pwd=6qje) (code: `6qje`) / [Google Drive](https://drive.google.com/file/d/1qNy-BcLAdOmRYb9q-qX_2vrfDOuudV0a/view?usp=drive_link) |

## Documentation

- [Installation](docs/Installation.md)
- [Dataset](docs/Dataset.md)
- [Training, Evaluation, and Test](docs/train_eval_test.md)

## Quick Start

### 1. Create the environment

Please follow the full setup instructions in [docs/Installation.md](docs/Installation.md).

Core environment:

```bash
conda create -n ffps python=3.7 -y
conda activate ffps
```

### 2. Download the pretrained EfficientNet-B7 backbone

The EfficientNet-B7 configs expect this checkpoint in `ckpts/`:

```text
ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth
```

Downloads:
- [Baidu Pan](https://pan.baidu.com/s/11OpePwIcKBkiONxiFK832Q?pwd=5pve) (code: `5pve`)
- [Google Drive](https://drive.google.com/file/d/10QqaKuXwV5JiVrqXCAW5mmUCr3J3XSgF/view?usp=drive_link)

### 3. Prepare datasets

Dataset preparation instructions are available in [docs/Dataset.md](docs/Dataset.md).

Supported datasets:

- SemanticKITTI
- SSCBench-KITTI360

For SemanticKITTI preprocessing, run:

```bash
bash process_kitti.sh
```

### 4. Train the model

SemanticKITTI:

```bash
python tools/train.py projects/configs/ffps_kitti.py --launcher none
```

SSCBench-KITTI360:

```bash
python tools/train.py projects/configs/ffps_sscbench_samearch.py --launcher none
```

More training, validation, test, and submission commands are listed in [docs/train_eval_test.md](docs/train_eval_test.md).

## Visualization

This repository provides two visualization modes for SemanticKITTI:

- Official-style GT visualization for SemanticKITTI `.label` voxel annotations
- Mayavi-based rendering for FFPS prediction `.pkl` files

Visualization tools are located in [tools/visualization](tools/visualization/README.md).

We recommend a separate visualization environment because the rendering stack differs from the training environment.

```bash
conda create -n ffps-vis python=3.9 -y
conda activate ffps-vis
pip install -r tools/visualization/requirements-visualization.txt
```

Examples:

```bash
python tools/visualization/view_semkitti_gt_label.py \
  --dataset data/SemanticKITTI/dataset \
  --sequence 08 \
  --frame 000645
```

```bash
python tools/visualization/visualize_semkitti_pred_pkl_mayavi.py \
  path/to/000645.pkl \
  --dataset semantic-kitti \
  --calib data/SemanticKITTI/dataset/sequences/08/calib.txt \
  --show
```

## Repository Note

The model name in this repository is `FFPS`. Some internal package directories still use the historical path `projects/mmdet3d_plugin/soap/` for compatibility with the existing codebase.
