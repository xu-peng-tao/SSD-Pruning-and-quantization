# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/6/9
import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
def obtain_avg_forward_time(input, model, repeat=200):  #repeat=200准确

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time

def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights

def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            if module_def['type']=='convolutional':
                if module_def['filters']=='-1':
                    break  #如果是-1，就不写了
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)
    if module_defs[idx-1]['type']=='maxpool':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    # for mobilenet
    elif module_defs[idx - 1]['type'] == 'se':
        return CBLidx2mask[idx - 3]
    elif module_defs[idx - 1]['type'] == 'depthwise':#depthwise不减
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]

def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask,prune_after):
    compact_predictor = compact_model.box_head.predictor
    loose_predictor = loose_model.box_head.predictor
    #print(compact_model.backbone.l2_norm)  没有l2时，是为0
    if compact_model.backbone.l2_norm !=0:
        compact_model.backbone.l2_norm.weight.data=loose_model.backbone.l2_norm.weight.data.clone()
    for idx in CBL_idx:
        if prune_after+1==idx and prune_after!=-1: # 这一层连同之后都不要
            break
        compact_CBL = compact_model.backbone.module_list[idx]
        loose_CBL = loose_model.backbone.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()  #非零

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.backbone.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

        if loose_conv.bias is not None:
            compact_conv.bias.data = loose_conv.bias.data[out_channel_idx].clone()


        if loose_model.backbone.module_defs[idx]['feature'] == 'linear':
            input_mask = get_input_mask(loose_model.backbone.module_defs, idx+1, CBLidx2mask)#input是上一层，这里加1
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            i = int(loose_model.backbone.module_defs[idx]['feature_idx'])
            # 定位
            loose_model_reg_conv = loose_predictor.reg_headers[i]
            compact_model_reg_conv = compact_predictor.reg_headers[i]
            if isinstance(loose_model_reg_conv,nn.Conv2d):
                compact_model_reg_conv.weight.data = loose_model_reg_conv.weight.data[:, in_channel_idx, :, :].clone()
                compact_model_reg_conv.bias.data = loose_model_reg_conv.bias.data.clone()
                # print(compact_model_reg_conv.weight.data.shape)
                # print(compact_model_reg_conv.bias.data.shape)
            else:#ssdlite深度可分离
                #深度可分离
                compact_model_reg_conv.conv[0].weight.data = loose_model_reg_conv.conv[0].weight.data[in_channel_idx, :, :, :].clone()

                #BN
                compact_model_reg_conv.conv[1].running_mean.data=loose_model_reg_conv.conv[1].running_mean.data[in_channel_idx].clone()
                compact_model_reg_conv.conv[1].running_var.data = loose_model_reg_conv.conv[
                    1].running_var.data[in_channel_idx].clone()
                compact_model_reg_conv.conv[1].weight.data = loose_model_reg_conv.conv[
                    1].weight.data[in_channel_idx].clone()
                compact_model_reg_conv.conv[1].bias.data = loose_model_reg_conv.conv[
                    1].bias.data[in_channel_idx].clone()
                #conv
                compact_model_reg_conv.conv[3].weight.data = loose_model_reg_conv.conv[3].weight[:, in_channel_idx, :, :].data.clone()
                compact_model_reg_conv.conv[3].bias.data = loose_model_reg_conv.conv[3].bias.data.clone()



            # 分类
            loose_model_cls_conv = loose_predictor.cls_headers[i]
            compact_model_cls_conv = compact_predictor.cls_headers[i]
            if isinstance(loose_model_cls_conv, nn.Conv2d):
                compact_model_cls_conv.weight.data = loose_model_cls_conv.weight.data[:, in_channel_idx, :, :].clone()
                compact_model_cls_conv.bias.data = loose_model_cls_conv.bias.data.clone()
            else:  # ssdlite深度可分离
                # 深度可分离
                compact_model_cls_conv.conv[0].weight.data = loose_model_cls_conv.conv[0].weight.data[in_channel_idx, :,
                                                             :, :].clone()

                # BN
                compact_model_cls_conv.conv[1].running_mean.data = loose_model_cls_conv.conv[1].running_mean.data[
                    in_channel_idx].clone()
                compact_model_cls_conv.conv[1].running_var.data = loose_model_cls_conv.conv[
                    1].running_var.data[in_channel_idx].clone()
                compact_model_cls_conv.conv[1].weight.data = loose_model_cls_conv.conv[
                    1].weight.data[in_channel_idx].clone()
                compact_model_cls_conv.conv[1].bias.data = loose_model_cls_conv.conv[
                    1].bias.data[in_channel_idx].clone()
                # conv
                compact_model_cls_conv.conv[3].weight.data = loose_model_cls_conv.conv[3].weight[:, in_channel_idx, :,
                                                             :].data.clone()
                compact_model_cls_conv.conv[3].bias.data = loose_model_cls_conv.conv[3].bias.data.clone()

        elif loose_model.backbone.module_defs[idx]['feature'] == 'l2_norm':
            i = int(loose_model.backbone.module_defs[idx]['feature_idx'])
            # 定位
            loose_model_reg_conv = loose_predictor.reg_headers[i]
            compact_model_reg_conv = compact_predictor.reg_headers[i]
            compact_model_reg_conv.weight.data = loose_model_reg_conv.weight.clone()
            compact_model_reg_conv.bias.data = loose_model_reg_conv.bias.data.clone()
            # print(compact_model_reg_conv.weight.data.shape)
            # print(compact_model_reg_conv.bias.data.shape)
            # 分类
            loose_model_cls_conv = loose_predictor.cls_headers[i]
            compact_model_cls_conv = compact_predictor.cls_headers[i]
            compact_model_cls_conv.weight.data = loose_model_cls_conv.weight.clone()
            compact_model_cls_conv.bias.data = loose_model_cls_conv.bias.data.clone()
        elif loose_model.backbone.module_defs[idx+1]['type']=='shortcut' and loose_model.backbone.module_defs[idx+1]['feature']=='linear':
            input_mask = get_input_mask(loose_model.backbone.module_defs, idx+2 , CBLidx2mask)
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            i = int(loose_model.backbone.module_defs[idx+1]['feature_idx'])
            # 定位

            loose_model_reg_conv = loose_predictor.reg_headers[i]
            compact_model_reg_conv = compact_predictor.reg_headers[i]


            if isinstance(loose_model_reg_conv, nn.Conv2d):
                compact_model_reg_conv.weight.data = loose_model_reg_conv.weight.data[:, in_channel_idx, :, :].clone()
                compact_model_reg_conv.bias.data = loose_model_reg_conv.bias.data.clone()
                # print(compact_model_reg_conv.weight.data.shape)
                # print(compact_model_reg_conv.bias.data.shape)
            else:  # ssdlite深度可分离
                # 深度可分离
                compact_model_reg_conv.conv[0].weight.data = loose_model_reg_conv.conv[0].weight.data[in_channel_idx, :,
                                                             :, :].clone()

                # BN
                compact_model_reg_conv.conv[1].running_mean.data = loose_model_reg_conv.conv[1].running_mean.data[
                    in_channel_idx].clone()
                compact_model_reg_conv.conv[1].running_var.data = loose_model_reg_conv.conv[
                    1].running_var.data[in_channel_idx].clone()
                compact_model_reg_conv.conv[1].weight.data = loose_model_reg_conv.conv[
                    1].weight.data[in_channel_idx].clone()
                compact_model_reg_conv.conv[1].bias.data = loose_model_reg_conv.conv[
                    1].bias.data[in_channel_idx].clone()
                # conv
                compact_model_reg_conv.conv[3].weight.data = loose_model_reg_conv.conv[3].weight[:, in_channel_idx, :,
                                                             :].data.clone()
                compact_model_reg_conv.conv[3].bias.data = loose_model_reg_conv.conv[3].bias.data.clone()

            # 分类
            loose_model_cls_conv = loose_predictor.cls_headers[i]
            compact_model_cls_conv = compact_predictor.cls_headers[i]
            if isinstance(loose_model_cls_conv, nn.Conv2d):
                compact_model_cls_conv.weight.data = loose_model_cls_conv.weight.data[:, in_channel_idx, :, :].clone()
                compact_model_cls_conv.bias.data = loose_model_cls_conv.bias.data.clone()
            else:  # ssdlite深度可分离
                # 深度可分离
                compact_model_cls_conv.conv[0].weight.data = loose_model_cls_conv.conv[0].weight.data[in_channel_idx, :,
                                                             :, :].clone()

                # BN
                compact_model_cls_conv.conv[1].running_mean.data = loose_model_cls_conv.conv[1].running_mean.data[
                    in_channel_idx].clone()
                compact_model_cls_conv.conv[1].running_var.data = loose_model_cls_conv.conv[
                    1].running_var.data[in_channel_idx].clone()
                compact_model_cls_conv.conv[1].weight.data = loose_model_cls_conv.conv[
                    1].weight.data[in_channel_idx].clone()
                compact_model_cls_conv.conv[1].bias.data = loose_model_cls_conv.conv[
                    1].bias.data[in_channel_idx].clone()
                # conv
                compact_model_cls_conv.conv[3].weight.data = loose_model_cls_conv.conv[3].weight[:, in_channel_idx, :,
                                                             :].data.clone()
                compact_model_cls_conv.conv[3].bias.data = loose_model_cls_conv.conv[3].bias.data.clone()
    for idx in Conv_idx:
        if prune_after+1<idx and prune_after!=-1: # 这一层连同之后都不要
            break
        compact_conv = compact_model.backbone.module_list[idx][0]
        loose_conv = loose_model.backbone.module_list[idx][0]


        input_mask = get_input_mask(loose_model.backbone.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        # 拷贝非剪植层的时候包括2种情况
        # 情况1：卷积层，需要拷贝bias
        # 情况2：depthwise层，直接拷贝卷积weight（它前面的未剪枝），还有bn参数
        if loose_model.backbone.module_defs[idx]['type'] == 'convolutional':
            compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.bias.data = loose_conv.bias.data.clone()
        else:
            compact_conv.weight.data = loose_conv.weight.data.clone()
            compact_bn = compact_model.backbone.module_list[idx][1]
            loose_bn = loose_model.backbone.module_list[idx][1]
            compact_bn.weight.data = loose_bn.weight.data.clone()
            compact_bn.bias.data = loose_bn.bias.data.clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data.clone()
            compact_bn.running_var.data = loose_bn.running_var.data.clone()

