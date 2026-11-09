# SemanticKITTI: bash tools/dist_train.sh projects/configs/ffps_kitti.py 4
# SemanticKITTI (ResNet3D flow): bash tools/dist_train.sh projects/configs/ffps_resnet_kitti.py 4
# SSCBench-KITTI360: bash tools/dist_train.sh projects/configs/ffps_sscbench_samearch.py 4
# SSCBench benchmark: bash tools/dist_train.sh projects/configs/ffps_bench.py 4
CONFIG=$1 
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3} \
    # --resume-from resued_param.pth

