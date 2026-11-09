## Training, Validation, and Test

All commands below are executed from the repository root.

### 1. SemanticKITTI

#### Training
Single-GPU:
```bash
python tools/train.py projects/configs/ffps_kitti.py --launcher none
```

Multi-GPU:
```bash
bash tools/dist_train.sh projects/configs/ffps_kitti.py 4
```

#### Validation
The validation split for SemanticKITTI is sequence `08`.

Single-GPU:
```bash
python tools/test.py \
  projects/configs/ffps_kitti.py \
  work_dirs/your_exp/latest.pth \
  --launcher none \
  --eval bbox \
  --cfg-options data.test.split=val
```

Multi-GPU:
```bash
bash tools/dist_test.sh \
  projects/configs/ffps_kitti.py \
  work_dirs/your_exp/latest.pth \
  4 \
  --cfg-options data.test.split=val
```

Optional: save validation predictions for visualization.
```bash
python tools/test.py \
  projects/configs/ffps_kitti.py \
  work_dirs/your_exp/latest.pth \
  --launcher none \
  --eval bbox \
  --pred-save work_dirs/semkitti_val_preds/ffps \
  --cfg-options data.test.split=val
```

#### Test Submission
SemanticKITTI does not provide public ground truth for the test split. Therefore, test performance must be evaluated by generating submission files and uploading them to:

`https://www.codabench.org/competitions/13814/`

Generate test predictions:
```bash
python tools/test.py \
  projects/configs/ffps_kitti.py \
  work_dirs/your_exp/latest.pth \
  --launcher none \
  --test-save work_dirs/semkitti_submit/ffps
```

This command writes predictions in the required structure:
```text
work_dirs/semkitti_submit/ffps/
└── sequences/
    ├── 11/
    │   └── predictions/
    │       ├── 000000.label
    │       ├── 000001.label
    │       └── ...
    ├── 12/
    │   └── predictions/
    ├── ...
    └── 21/
        └── predictions/
```

Compress the generated `sequences` directory and submit it on CodaBench.

### 2. SSCBench-KITTI360

#### Training
Single-GPU:
```bash
python tools/train.py projects/configs/ffps_sscbench_samearch.py --launcher none
```

Multi-GPU:
```bash
bash tools/dist_train.sh projects/configs/ffps_sscbench_samearch.py 4
```

#### Validation
The validation split is `2013_05_28_drive_0006_sync`.

Single-GPU:
```bash
python tools/test.py \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  --launcher none \
  --eval bbox \
  --cfg-options data.test.split=val
```

Multi-GPU:
```bash
bash tools/dist_test.sh \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  4 \
  --cfg-options data.test.split=val
```

Optional: save validation predictions for visualization.
```bash
python tools/test.py \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  --launcher none \
  --eval bbox \
  --pred-save work_dirs/ffps_sscbench_samearch/val_preds \
  --cfg-options data.test.split=val
```

#### Test
Unlike SemanticKITTI, SSCBench-KITTI360 provides ground truth for its test split in this project setup, so you can directly run evaluation on the test set.

Single-GPU:
```bash
python tools/test.py \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  --launcher none \
  --eval bbox
```

Multi-GPU:
```bash
bash tools/dist_test.sh \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  4
```

Optional: save test predictions.
```bash
python tools/test.py \
  projects/configs/ffps_sscbench_samearch.py \
  work_dirs/ffps_sscbench_samearch/latest.pth \
  --launcher none \
  --eval bbox \
  --pred-save work_dirs/ffps_sscbench_samearch/test_preds
```

