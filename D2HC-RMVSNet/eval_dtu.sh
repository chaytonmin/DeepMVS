#!/usr/bin/env bash
source env.sh
model=drmvsnet
n=1
batch=1
d=512
interval_scale=0.4
data=$(date +"%m%d")
batch=1
lr=0.001
lr_scheduler=cosinedecay
loss=mvsnet_loss_l1norm
fea_net=FeatNet
cost_net=UNetConvLSTM
inverse_depth=False
syncbn=False
origin_size=False
refine=False
save_depth=True
fusion=False
gn=True
checkpoint_list=(04)
gpu_list=(0 1 2)
ckpt=./checkpoints/model_0000
now=$(date +"%Y%m%d_%H%M%S")
echo $now
for idx in ${checkpoint_list[@]}
do
name=testPoint_${data}_g${n}_b${batch}_${model}_${fea_net}_${cost_net}_id${inverse_depth}_sd${save_depth}_fu${fusion}_ckpt${idx}
echo $name
echo 'process light'${light_idx}
CUDA_VISIBLE_DEVICES=2 python -u eval.py \
        --dataset=data_eval_transform \
        --model=$model \
        --syncbn=$syncbn \
        --batch_size=${batch} \
        --inverse_cost_volume \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --gn=$gn \
        --refine=$refine \
        --save_depth=$save_depth \
        --fusion=False \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --max_h=360 \
        --max_w=480 \
        --image_scale=1.0 \
        --pyramid=0 \
        --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --loadckpt=$ckpt${idx}.ckpt \
        --outdir=./outputs_1032
done
