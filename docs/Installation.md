## Installation

The current FFPS repository is tested with Python 3.7, PyTorch 1.10.1, and CUDA 11.3.
All commands below are executed from the repository root.

### 1. Create a conda virtual environment
```shell
conda create -n ffps python=3.7 -y
conda activate ffps
```

### 2. Install PyTorch
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Or:
```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3. Install MMCV / MMDetection / MMSegmentation
```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

### 4. Install the local MMDetection3D package
This repository keeps a local `mmdetection3d` source tree with custom operators required by the model.

```shell
cd mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
```

### 5. Build Multi-Scale Deformable Attention
```shell
cd projects/mmdet3d_plugin/soap/necks/ops
python setup.py install
cd ../../../../..
```

### 6. Build the D3D deformable convolution extension
```shell
cd projects/mmdet3d_plugin/soap/dcn
python setup.py install
cd ../../../..
```

### 7. Install additional Python dependencies used by FFPS
```shell
pip install timm open3d-python easydict fvcore einops torchmetrics setuptools==59.5.0 yapf==0.40.1
```

### 8. Install preprocessing dependencies for `process_kitti.sh`
```shell
pip install tqdm pyyaml imageio
```

### 9. Download pretrained backbone weights
The EfficientNet-B7 configs in this repository expect the pretrained backbone checkpoint at:

```text
ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth
```

Download it and place it in the `ckpts/` directory:

- Baidu Pan: https://pan.baidu.com/s/11OpePwIcKBkiONxiFK832Q?pwd=5pve
  Code: `5pve`
- Google Drive: https://drive.google.com/file/d/10QqaKuXwV5JiVrqXCAW5mmUCr3J3XSgF/view?usp=drive_link

### 10. Notes
- If you only train or evaluate the model, Step 8 is only needed when you want to run `bash process_kitti.sh` for SemanticKITTI label preprocessing.
- If `open3d-python` fails in your environment, try `pip install open3d` instead.
- The SemanticKITTI preprocessing script assumes data under `data/SemanticKITTI`, and writes label files to `data/SemanticKITTI/dataset/labels`.
- If you use the ResNet-based configs, download the corresponding ResNet-50 pretrained checkpoint separately and place it under `ckpts/`.
