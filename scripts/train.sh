#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=4,5
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:`pwd`


config_path='baseline.MetaRS'
model_dir='./log/EarthMiss/MetaRS'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 1245 --use_env train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --trainer=th_ddp \
    train.eval_interval_epoch 20\
    train.save_ckpt_interval_epoch 20\
    data.train.params.batch_size  4 \
    data.train.params.CV.cur_k -1 \
    data.test.params.CV.cur_k -1 \
    train.num_iters 15000 \
    learning_rate.params.max_iters 15000  \
    learning_rate.params.base_lr 1e-4 \
    