# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/6/30
import torch
import  os
import argparse
from ssd.config import cfg
from torchtoolbox.tools import summary
from ssd.modeling.detector import build_detection_model
#只计算backbone
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='SSD FLOPs')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--in_size",
        default=300,
        help="input size",
        type=int,
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
    cfg.freeze()
    Model = build_detection_model(cfg).backbone
    summary(Model,torch.rand((1,3,args.in_size,args.in_size)))

if __name__ == '__main__':
    main()