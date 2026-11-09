cd $(readlink -f `dirname $0`)
# Run from the repository root so relative paths below stay valid.
# Activate your environment before execution if needed.

export PYTHONPATH="."
export OMP_NUM_THREADS=8

kitti_root=./data/SemanticKITTI
kitti_preprocess_root=./data/SemanticKITTI/dataset
data_info_path=./tools/data_converter/kitti_process/semantic-kitti.yaml

python ./tools/data_converter/kitti_process/semantic_kitti_preprocess.py \
--kitti_root $kitti_root \
--kitti_preprocess_root $kitti_preprocess_root \
--data_info_path $data_info_path
