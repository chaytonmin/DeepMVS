#!/usr/bin/env bash
source env.sh
model=drmvsnet
n=1
batch=1
d=512
interval_scale=0.5
data=$(date +"%m%d")
batch=1
lr=0.001
lr_scheduler=cosinedecay
loss=mvsnet_loss_l1norm
fea_net=FeatNet
cost_net=UNetConvLSTM
inverse_depth=True
syncbn=False
origin_size=False
refine=False
save_depth=True
return_depth=True
reg_loss=True
fusion=False
gn=True
checkpoint_list=(04)
ckpt=./checkpoints/model_0000
now=$(date +"%Y%m%d_%H%M%S")
echo $now
for idx in ${checkpoint_list[@]}
do
name=testPoint_${data}_g${n}_b${batch}_${model}_${fea_net}_${cost_net}_id${inverse_depth}_sd${save_depth}_fu${fusion}_ckpt${idx}
echo $name
echo 'process light'${light_idx}
CUDA_VISIBLE_DEVICES=1 python -u eval.py \
        --dataset=data_eval_transform_large \
        --model=$model \
        --syncbn=$syncbn \
        --batch_size=1 \
        --inverse_cost_volume \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --refine=$refine \
        --save_depth=$save_depth \
        --fusion=False \
        --gn=$gn \
        --return_depth=${return_depth} \
        --reg_loss=$reg_loss \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --max_h=512 \
        --max_w=960 \
        --image_scale=1.0 \
        --img_scale=2.0 \
        --pyramid=0 \
        --testpath=$TP_TESTING \
        --testlist=lists/tp_list.txt \
        --loadckpt=$ckpt${idx}.ckpt \
        --outdir=./outputs_tnt_1101 &
done
