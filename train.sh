#!/usr/bin/env bash
python setup.py install

python experiments/segmentation/train.py \
--dataset pcontext --model multi_nl_fcn \
--data-root /home/cola/.encoding/data/PASCAL_CONTEXT \
--checkname multi_nonlocal_aggre_region_1x2x_r101 \
--backbone resnet101 --epochs 50 --batch-size 16 --lr 0.001

python experiments/segmentation/test.py \
--dataset pcontext --model multi_nl_fcn \
--data-root /home/cola/.encoding/data/PASCAL_CONTEXT \
--backbone resnet101 \
--resume ./runs/pcontext/multi_nl_fcn/multi_nonlocal_aggre_region_1x2x_r101/model_best.pth.tar \
--eval

#CUDA_VISIBLE_DEVICES=0 python experiments/segmentation/visualize.py \
#--dataset pcontext --model multi_nl_fcn \
#--checkname multi_nonlocal_attn2scale_add_deconv_region_1x2x \
#--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
#--batch-size 1 \
#--backbone resnet50 --resume ./runs/pcontext/multi_nl_fcn/multi_nonlocal_attn2scale_add_deconv_region_1x2x/model_best.pth.tar