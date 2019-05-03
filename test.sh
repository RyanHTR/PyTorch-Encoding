#!/usr/bin/env bash
python setup.py install

python experiments/segmentation/test.py \
--dataset PContext --model multi_nl_fcn \
--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
--backbone resnet50 --resume ./runs/pcontext/multi_nl_fcn/multi_nonlocal_1x/model_best.pth.tar \
--eval