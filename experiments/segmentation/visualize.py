import torch
import encoding
from encoding.datasets import get_segmentation_dataset, test_batchify_fn, get_dataset
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
from encoding.nn import SegmentationLosses, SyncBatchNorm
import os
from torch.utils import data
import torchvision.transforms as transform
from tqdm import tqdm
import torchvision.utils as utils

from option import Options
from PIL import Image
import numpy as np


def visualize(args):
    directory = "runs/%s/%s/%s/vis/results"%(args.dataset, args.model, args.checkname)
    print("visualize directory : ", directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Get the model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux,
                                   se_loss=args.se_loss, norm_layer=SyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size)
    model = model.cuda()

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'])
    # for key in checkpoint['state_dict']:
    #     print(key)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    model.eval()


    # Prepare the image
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}

    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size, 'root': args.data_root}

    testset = get_dataset(args.dataset, split='val', mode='val',
                          **data_kwargs)

    testloader = data.DataLoader(testset, batch_size=1,
                                     drop_last=False, shuffle=False, **kwargs)

    avg = [.485, .456, .406]
    std = [.229, .224, .225]
    # visualize

    cnt = 0
    for i, (image, dst) in enumerate(tqdm(testloader)):
        if cnt == 100:
            break
        prob = np.random.rand(1)[0]
        if prob > 0.2:
            continue

        image = image.cuda()
        output = model.evaluate(image)
        pred = torch.max(output, 1)[1].cpu().numpy() + 1
        mask = encoding.utils.get_mask_pallete(pred, 'pascal_context')

        dst = dst.numpy() + 1
        gt = encoding.utils.get_mask_pallete(dst, 'pascal_context')

        im = image.cpu().numpy().squeeze().transpose(1, 2, 0)
        im = im * std + avg
        im = im * 255
        im = im.astype('uint8')
        im = Image.fromarray(im)

        target = Image.new('RGB', (480*3 + 20, 480), color=(255, 255, 255))
        target.paste(im, (0, 0))
        target.paste(gt, (490, 0))
        target.paste(mask, (980, 0))
        target.save('{}/{}.png'.format(directory, str(i)))
        cnt += 1


def visualize_attn(args):
    directory = "runs/%s/%s/%s/vis/attn" % (args.dataset, args.model, args.checkname)
    print("visualize directory : ", directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Get the model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux,
                                   se_loss=args.se_loss, norm_layer=SyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size)
    model = model.cuda()
    # print(model)
    # print("=================================")
    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'])
    # for key in checkpoint['state_dict']:
    #     print(key)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    model.eval()

    # Prepare the image
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}

    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size, 'root': args.data_root}

    testset = get_dataset(args.dataset, split='val', mode='val',
                          **data_kwargs)

    testloader = data.DataLoader(testset, batch_size=1,
                                 drop_last=False, shuffle=False, **kwargs)

    avg = [.485, .456, .406]
    std = [.229, .224, .225]
    # visualize

    cnt = 0
    for i, (image, dst) in enumerate(tqdm(testloader)):
        if cnt == 100:
            break
        prob = np.random.rand(1)[0]
        if prob > 0.2:
            continue

        image = image.cuda()
        output = model.evaluate(image)
        cnt += 1
        print("{}/100".format(cnt))


if __name__ == '__main__':
    args = Options().parse()
    args.test_batch_size = 1
    visualize(args)
    # visualize_attn(args)
