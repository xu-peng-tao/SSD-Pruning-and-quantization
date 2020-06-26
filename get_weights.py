# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/15

import torch
import  os
import numpy as np
import sys
import argparse
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='SSD WEIGHTS')
    parser.add_argument(
        "--model",
        default='no',
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        default='model_final.pth',
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    name = cfg.OUTPUT_DIR.split('/')[1]
    print(name)
    if args.model != 'no':
        model_path = args.model
    else:
        model_path = 'outputs/' + name + '/' + args.ckpt

    np.set_printoptions(threshold=sys.maxsize)  # 全部输出,无省略号
    np.set_printoptions(suppress=True)  # 不用指数e
    state = torch.load(model_path, map_location=torch.device('cpu'))    # print(state['model'])
    file = open('weights/' + name + '_para.txt', 'w')
    if "model" in state.keys():
        model = state['model']
    else:
        model=state
    if cfg.TEST.BN_FUSE is True:
        print('BN_FUSE.')
        cfg.MODEL.BACKBONE.PRETRAINED = False
        Model = build_detection_model(cfg)
        # print(Model)
        Model.load_state_dict(model)
        Model.backbone.bn_fuse()
        model=Model.state_dict()
    for name in model:
        print(name)
        para = model[name]
        print(para.shape)
        file.write(str(name) + ':\n')
        file.write('shape:' + str(para.shape) + '\n')
        file.write('para:\n' + str(para.cpu().data.numpy()) + '\n')
    file.close()


if __name__ == '__main__':
    main()
