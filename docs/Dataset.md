## Prepare Dataset

### a. SemanticKITTI
To prepare the SemanticKITTI dataset, first download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color and calibration files), along with the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download).

1. Create stereo depth maps by following [image2depth_semantickitti.sh](https://github.com/gywns6287/SOAP/blob/main/preprocess/image2depth_semantickitti.sh).
2. Generate voxel label `.npy` files from the SemanticKITTI voxel annotations by running the following command from the repository root:

```bash
bash process_kitti.sh
```

Place all `.zip` files under `FFPS/data/SemanticKITTI` and unzip them. After extraction and preprocessing, your dataset directory should look like this:

```
FFPS
├── data/
│   ├── SemanticKITTI/
│   │   ├── dataset/
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   ├── image_2/
│   │   │   │   │   ├── voxels/
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── ...
│   │   │   │   ├── 21
│   │   │   ├── stereodepth/
│   │   │   │   ├── 00
│   │   │   │   │   ├── depth/
│   │   │   │   │   │   ├── 000000.npy
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   │   ├── 21
│   │   │   ├── labels/
│   │   │   │   ├── 00
│   │   │   │   │   ├── 000000_1_1.npy
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   │   ├── 10
```

### b. SSCBench-KITTI360 
Download the dataset from [SSCBench-KITTI-360](https://github.com/ai4ce/SSCBench), and prepare the depth maps by following the preprocessing steps from [image2depth_kitti360.sh](https://github.com/gywns6287/SOAP/blob/main/preprocess/image2depth_kitti360.sh).  The pose data for SSCBench-KITTI360 can be downloaded [here](https://drive.google.com/file/d/1bn5aEs-Hx6w2YE7VLQRAZCCF2qJegW6E/view?usp=sharing
). After setup, your dataset directory should look like this:
```
FFPS
├── data/
│   ├── SSCBench-KITTI360/
│   │   ├── data_2d_raw/
│   │   │   ├── 2013_05_28_drive_0000_sync/
|   │   │   │   ├── image_00/
|   │   │   │   ├── voxels/
|   │   │   │   ├── poses.txt
│   │   │   ├── 2013_05_28_drive_0002_sync/
│   │   │   ├── ...
│   │   │   ├── 2013_05_28_drive_0010_sync/
│   │   ├── stereodepth_bench/
│   │   │   ├── 2013_05_28_drive_0000_sync/
|   │   │   │   ├── depth/
│   │   │   │   │   ├── 000000.npy
│   │   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── 2013_05_28_drive_0010_sync/

```
