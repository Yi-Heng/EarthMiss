
export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

modality="SAR"
ckpt_path="./Best2.pth"
config_path="./configs/baseline/MetaRS.py"

vis_path="vis-$(basename ${ckpt_path} .pth)"
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 1060 --use_env eval.py \
    --ckpt_path=${ckpt_path} \
    --config_path=${config_path} \
    --vis_path=${vis_path} \
    --modality=${modality}



