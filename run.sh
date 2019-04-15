#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/segmentation/train.py \
--dataset citys --model fcn \
--dataroot /home/htr/DATASET/cityscapes \
--backbone resnet101 --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003