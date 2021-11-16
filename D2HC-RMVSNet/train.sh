#!/bin/bash
source env.sh
data=$(date +"%m%d")
model=drmvsnet
n=4
batch=1
epochs=10
d=192
interval_scale=1.06
lr=0.001
lr_scheduler=cosinedecay
loss=mvsnet_cls_loss_ori
optimizer=Adam
fea_net=FeatNet
cost_net=UNetConvLSTM
loss_w=4
inverse_depth=False
image_scale=0.25
view_num=7
gn=True
name=${data}_drmvs_g${n}_b${batch}_d${d}_is${interval_scale}_${loss}_lr${lr}_op${optimizer}_ep${epochs}_${lrepochs}_sh${lr_scheduler}_${fea_net}_${cost_net}_vn${view_num}_is${image_scale}_gn${gn}_v1.26_newdepth
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$n --master_port 10190 train.py  \
        --dataset=dtu_yao \
        --model=$model \
        --batch_size=${batch} \
        --trainpath=$MVS_TRAINING \
        --loss=${loss} \
        --lr=${lr} \
        --epochs=${epochs} \
        --loss_w=$loss_w \
        --lr_scheduler=$lr_scheduler \
        --optimizer=$optimizer \
        --view_num=$view_num \
        --inverse_depth=${inverse_depth} \
        --image_scale=$image_scale \
        --using_apex \
        --gn=$gn \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --trainlist=lists/dtu/train.txt \
        --vallist=lists/dtu/val.txt \
        --testlist=lists/dtu/test.txt \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./checkpoints/${name} \
        2>&1|tee ./logs/${name}-${now}.log &
