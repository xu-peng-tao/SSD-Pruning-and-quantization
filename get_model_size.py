# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/4/25
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
import os
obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])
def model_size(model):#暂时不能处理最后层量化情况
    backbone=model.backbone
    # print(backbone)
    count_qw=0#算出量化的参数个数
    qw_size=0#量化的参数size  Byte
    for i in range(len(backbone.module_list)):
        # print(backbone.module_defs[i])
        if backbone.module_defs[i]["type"]=='convolutional':
            if ("quantization" in backbone.module_defs[i].keys()) and (backbone.module_defs[i]["quantization"]=='1'):
                conv = backbone.module_list[i][0]
                W_bits=conv.w_bits
                W_B=W_bits/8  #Byte=bit/8
                qw_size+=W_B*conv.weight.data.flatten().shape[0]
                count_qw+=conv.weight.data.flatten().shape[0]
    model_size=(obtain_num_parameters(model)-count_qw)*4+qw_size

    model_size=model_size/1024.0/1024.0
    return model_size


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cfg_path="configs/vgg_bn_ssd300_hand_fpga_cutPredict.yaml"
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.BACKBONE.PRETRAINED = False
    model=build_detection_model(cfg)
    print(model)
    print(f'model_paras:{obtain_num_parameters(model)/1024.0/1024.0}M')
    size=model_size(model=model)
    print(f'model_size:{size}MB')