#!/usr/bin/env bash
python setup.py install

python experiments/segmentation/test.py \
--dataset PContext --model multi_nl_fcn \
--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
--backbone resnet50 --resume ./runs/pcontext/fcn/multi_nonlocal_1x2x/model_best.pth.tar \
--eval