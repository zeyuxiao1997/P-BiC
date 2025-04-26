import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from functools import partial
import time

def getNetworkDescription(network):
    # if isinstance(network, nn.DataParallel) or isinstance(network, nn.DistributedDataParallel):
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n

def model_fn_decorator(loss_fn, mode='train'):
    def test_model_fn(args, data, model, save_path, compute_metrics):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0
        # print(data.keys())
        in_img = data['in_img'].cuda()
        moire_green = data['moire_green'].cuda()
        label = data['label'].cuda()
        b, c, h, w = in_img.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        moire_green = img_pad(moire_green, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        # model.eval().cuda()
        with torch.no_grad():
            st = time.time()
            out_1, out_2, out_3 = model(in_img, moire_green)
            cur_time = time.time()-st
            if h_pad != 0:
               out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
               out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

        if args.EVALUATION_METRIC:
            cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out_1, label)

        # save images
        if args.SAVE_IMG:
            out_save = out_1.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)

        return cur_psnr, cur_ssim, cur_lpips, cur_time

    def model_fn(args, data, model, iters):
        # _, n = getNetworkDescription(model)
        # print('params: ',n)
        model.train().cuda()
        # prepare input and forward
        in_img = data['in_img'].cuda()
        label = data['label'].cuda()
        moire_green = data['moire_green'].cuda()
        out_1, out_2, out_3 = model(in_img, moire_green)
        loss = loss_fn(out_1, out_2, out_3, label)

        # save images
        if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            in_save = in_img.detach().cpu()
            out_save = out_1.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((in_save, out_save, gt_save), 3)
            save_number = (iters + 1) // args.SAVE_ITER
            torchvision.utils.save_image(res_save,
                                         args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')


        return loss

    if mode == 'test':
        fn = test_model_fn
    else:
        fn = model_fn
    return fn