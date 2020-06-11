# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/15

import torch
from ssd.modeling.detector import build_detection_model
from ssd.config import cfg


config_file= 'configs/mobile_v2_ssd_hand_normal_sparse.yaml'  #'configs/vgg_ssd300_hand_fpga.yaml'
cfg.merge_from_file(config_file)
cfg.freeze()
model=build_detection_model(cfg)
print(model)
