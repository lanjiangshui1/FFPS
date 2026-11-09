# SSCBench benchmark: bash tools/dist_test.sh projects/configs/ffps_bench.py ffps_bench.pth 4
# SemanticKITTI: bash tools/dist_test.sh projects/configs/ffps_kitti.py ffps_kitti.pth 4
# SemanticKITTI (ResNet3D flow): bash tools/dist_test.sh projects/configs/ffps_resnet_kitti.py ffps_resnet_kitti.pth 4
# SSCBench-KITTI360: bash tools/dist_test.sh projects/configs/ffps_sscbench_samearch.py ffps_sscbench.pth 4

CONFIG=$1 
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox \
    # --test-save=results # trun off to evaluation


