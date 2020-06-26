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
#从fpga的量化权重得到pytorch权重文件
#需要修改demo.py   要bn_fuse   因为fpga端没有bn
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='SSD WEIGHTS')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        default='fpga/test.pth',
        type=str,
    )
    parser.add_argument(
        "--fpga",
        default='fpga/',
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
    cfg.freeze()

    b_file = open('fpga/' + 'mini_ssd_hand_fpga_bias_q.txt', 'r')
    # b = b_file.readline().rstrip('\n')
    # print(float(b))

    w_file = open('fpga/' + 'mini_ssd_hand_fpga_weights_q.txt', 'r')
    # w=w_file.readline().rstrip('\n')
    # print(float(w))



    Model = build_detection_model(cfg)
    # print(Model)
    Model.backbone.bn_fuse()
    print(Model)
    model=Model.state_dict()
    # print(model)

    for name in model:
        print(name)
        shape = model[name].shape
        print(shape)
        if name.find('weight')>=0:
            file = w_file
            print('weight')
        else:
            file=b_file
            print('bias')
        # fpga_mod = torch.flatten(model[name]).numpy()
        length=1
        for i in range(len(shape)):
            length=shape[i]*length
        # print(length)
        fpga_mod=np.zeros(length)  #全零，等待获取权重
        for i in range(length):
            fpga_mod[i]=float(file.readline().rstrip('\n'))
        # print(fpga_mod)

        model[name]=torch.reshape(torch.tensor(fpga_mod,dtype=model[name].dtype),shape)
        # print(model[name])

    torch.save(model,args.ckpt)
    w_file.close()
    b_file.close()

if __name__ == '__main__':
    main()
