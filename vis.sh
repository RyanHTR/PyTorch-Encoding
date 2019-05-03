#!/usr/bin/env bash
#!/usr/bin/env bash
python setup.py install

CUDA_VISIBLE_DEVICES=0 python experiments/segmentation/visualize.py \
--dataset pcontext --model multi_nl_fcn \
--checkname multi_nonlocal_1x2x4x_attn2scale_deconv \
--data-root /home/htr/.encoding/data/PASCAL_CONTEXT \
--batch-size 2 \
--backbone resnet50 --resume ./runs/pcontext/multi_nl_fcn/multi_nonlocal_1x2x4x_attn2scale_deconv/model_best.pth.tar