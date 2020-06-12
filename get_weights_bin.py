# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/28


# import numpy as np
# import struct
# import os
#
# a=np.array([[1.2345 ,1.5698],[1.5888,1.6986]],dtype=np.float32)
# print(a)
#
# file = open('weights/'  + 'para.bin', 'wb')
# print(a.flatten())
# for i in a.flatten():
#     print(i)
#     a = struct.pack('>f', i)#大端，浮点
#     print(a)
#     file.write(a)
#
#
# file.close()
#
# filepath='weights/'  + 'para.bin'
# file=open(filepath, 'rb')
# size = os.path.getsize(filepath) #获得文件大小
# print(size)
# for i in range(int(size/4)):
#     data = file.read(4) #每次输出4个字节
#     print(data)
#     data=struct.unpack('>f', data)#解析
#     print(data)
# file.close()

import struct
import torch
import  os
import numpy as np
import sys
import argparse
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
#python get_weights_bin.py --config-file configs/vgg_bn_ssd300_hand_fpga.yaml --ckpt model_final.pth TEST.BN_FUSE True

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
        default='model_final.pth',
        type=str,
    )
    parser.add_argument(
        "--model",
        default='no',
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
    if args.model!='no':
        model_path=args.model
    else:
        model_path = 'outputs/'+name+'/'+args.ckpt

    state = torch.load(model_path, map_location=torch.device('cpu'))
    # print(state['model'])
    w_file = open('weights/' + name + '_weights.bin', 'wb')
    b_file=open('weights/'+name+'_bias.bin','wb')
    if "model" in state.keys():
        model = state['model']
    else:
        model=state
    if cfg.TEST.BN_FUSE is True:
        print('BN_FUSE.')
        cfg.MODEL.BACKBONE.PRETRAINED=False
        Model = build_detection_model(cfg)
        Model.load_state_dict(model)
        Model.backbone.bn_fuse()
        # print(Model)
        model=Model.state_dict()
    for name in model:
        print(name)
        para = model[name]
        print(para.shape)
        para_flatten=para.cpu().data.numpy().flatten()  #展开
        if name.find('weight')>=0:
            file = w_file
            print('weight')
        else:
            file=b_file
            print('bias')

        for i in para_flatten:
            # print(i)
            a = struct.pack('<f', i)# 小端浮点                 大端，浮点32>f
            # print(a)
            file.write(a)

    w_file.close()
    b_file.close()

if __name__ == '__main__':
    main()
