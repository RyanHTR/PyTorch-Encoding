#!/usr/bin/env bash
python setup.py install

python experiments/segmentation/train.py \
--dataset pcontext --model multi_nl_fcn \
--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
--checkname multi_nonlocal_1x2x \
--backbone resnet50 --epochs 50 --batch-size 16 --lr 0.001

python experiments/segmentation/test.py \
--dataset PContext --model multi_nl_fcn \
--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
--backbone resnet50 --resume ./runs/pcontext/fcn/multi_nonlocal_1x2x/model_best.pth.tar \
--eval