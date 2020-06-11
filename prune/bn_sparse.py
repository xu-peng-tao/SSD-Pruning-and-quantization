# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/16
import torch

def updateBN( module_list, s, prune_idx):
    for idx in prune_idx:
        # Squential(Conv, BN, Activate)
        bn_module = module_list[idx][1]
        # print(bn_module.weight.grad)
        bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1