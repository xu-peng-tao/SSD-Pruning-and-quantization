# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/6/9
import torch.nn as nn
from copy import deepcopy
from ssd.modeling.detector import build_detection_model
from ssd.engine.inference import do_evaluation
from prune.prune_utils import *
from get_model_size import model_size
from terminaltables import AsciiTable

def parse_module_defs(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
        elif module_def['type'] == 'depthwise':
            Conv_idx.append(i)
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                shortcut_idx[i - 1] = identity_idx  #short_cut 匹配
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)
        # 上采样层前的卷积层不裁剪
        if module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)
        # 深度可分离卷积层的其前一层不剪  待改进
        if module_def['type'] == 'depthwise':
            ignore_idx.add(i - 1)
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all




def obtain_filters_mask(model, thresh, CBL_idx, prune_idx,predictor_channels,  max):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    prune_ = 0  # 为1时表示找到第一个BN偏置全零层   max时后面的层都不要
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            mask = bn_module.weight.data.abs().ge(thresh).float()
            mask_cnt = int(mask.cpu().numpy().sum())
            if max==1:#最大剪枝
                if mask_cnt==0:
                    prune_=1

            remain = mask_cnt
            if remain == 0:
                # print("Channels would be all pruned!") #当有一层被完全剪掉，yolo采用了阙值，不让这么剪，我这里可以保留这个一个最大的，其余剪掉
                # raise Exception
                remain = 1
                max = bn_module.weight.data.abs().max()
                mask = bn_module.weight.data.abs().ge(max).float()
            pruned = pruned + mask.shape[0] - remain

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())
        if model.module_defs[idx]['feature'] == 'linear':
            i = int(model.module_defs[idx]['feature_idx'])
            if prune_ == 1:
                predictor_channels[i] = -1  # 不要这个预测层了
            else:
                predictor_channels[i] = remain  # 预测层输入通道数改变
                    # print(f'remian{remain}')
        # 因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数   除去了l2层，也进行了防止一层被裁掉的操作
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')  # 对于最大剪枝，这里是不准的

    return num_filters, filters_mask, predictor_channels


def prune_and_eval(model, CBL_idx, CBLidx2mask,cfg):
    print(f'mAP of the original model is:')
    with torch.no_grad():
        eval = do_evaluation(cfg, model, distributed=False)
        print(eval[0]['metrics'])
    model_copy = deepcopy(model)
    for idx in CBL_idx:
        bn_module = model_copy.backbone.module_list[idx][1]
        mask = CBLidx2mask[idx].cuda()
        bn_module.weight.data.mul_(mask)
    print('快速看剪枝效果----》')
    print(f'mAP of the pruned model is:')
    with torch.no_grad():
        eval = do_evaluation(cfg, model_copy, distributed=False)
        print(eval[0]['metrics'])

def prune_model_keep_size(cfg,model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    backbone = pruned_model.backbone
    predictor = pruned_model.box_head.predictor
    activations = []
    for i, model_def in enumerate(backbone.module_defs):

        if model_def['type'] == 'convolutional' or model_def['type'] == 'depthwise':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = backbone.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                activate = backbone.module_list[i][-1]
                bn_bias = bn_module.bias.data
                next_idx = i + 1  # 补偿下一层
                if next_idx != len(backbone.module_defs):  # 不超出最后一层
                    if backbone.module_defs[next_idx]['type'] == 'convolutional':

                        next_conv = backbone.module_list[next_idx][0]
                        next_conv_weight = next_conv.weight.data
                        if isinstance(activate, nn.BatchNorm2d):
                            activation = (1 - mask) * bn_bias
                        else:
                            activation = activate(
                                (1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》未剪掉的为0，得保证0量化为0
                        # if backbone.module_defs[next_idx]['quantization'] == '1':
                        #     activation=quan_a(activation)
                        #     next_conv_weight=quan_w(next_conv_weight)
                        # 池化影响在求prune_idx时已解决
                        conv_sum = next_conv_weight.sum(dim=(2, 3))
                        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                        if next_idx in CBL_idx:  # 对于convolutionnal，如果有BN，则该层卷积层不使用bias
                            next_bn = backbone.module_list[next_idx][1]
                            next_bn.running_mean.data.sub_(offset)
                            # next_conv.bias.data.add_(offset)
                        else:  # 如果无BN，则使用bias             针对mobile这样做，容易扩展
                            next_conv.bias.data.add_(offset)
                        # 补偿预测层
                        if backbone.module_defs[i]['feature'] == 'linear':  # 补偿预测层  （预测层不减枝，对于ssdlite可能可以改进)
                            i = int(backbone.module_defs[i]['feature_idx'])
                            # 补偿定位
                            reg_conv = predictor.reg_headers[i]
                            if isinstance(activate, nn.BatchNorm2d):
                                activation = (1 - mask) * bn_bias
                            else:
                                activation = activate(
                                    (1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》为剪掉的为0，得保证0量化为0
                            if isinstance(reg_conv, nn.Conv2d):
                                next_conv_weight = reg_conv.weight.data
                                # if cfg.QUANTIZATION.FINAL == True:
                                #     activation = quan_a(activation)
                                #     next_conv_weight = quan_w(next_conv_weight)
                                conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                                reg_conv.bias.data.add_(offset)
                            else:  # ssdlite深度可分离
                                next_conv_weight = reg_conv.conv[0].weight.data  # 深度卷积
                                # if cfg.QUANTIZATION.FINAL == True:
                                #     activation = quan_a(activation)
                                #     next_conv_weight = quan_w(next_conv_weight)
                                conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                                offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                                reg_conv.conv[1].running_mean.data.sub_(offset)

                            # 补偿分类
                            cls_conv = predictor.cls_headers[i]
                            if isinstance(activate, nn.BatchNorm2d):
                                activation = (1 - mask) * bn_bias
                            else:
                                activation = activate(
                                    (1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》为剪掉的为0，得保证0量化为0
                            if isinstance(cls_conv, nn.Conv2d):  # 普通ssdlite
                                next_conv_weight = cls_conv.weight.data
                                # if cfg.QUANTIZATION.FINAL == True:
                                #     activation = quan_a(activation)
                                #     next_conv_weight = quan_w(next_conv_weight)
                                conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                                cls_conv.bias.data.add_(offset)
                            else:  # ssdlite深度可分离
                                next_conv_weight = cls_conv.conv[0].weight.data
                                # if cfg.QUANTIZATION.FINAL == True:
                                #     activation = quan_a(activation)
                                #     next_conv_weight = quan_w(next_conv_weight)
                                conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                                offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                                cls_conv.conv[1].running_mean.data.sub_(offset)

                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            next_idx = i + 1  # 补偿下一层
            if next_idx != len(backbone.module_defs):  # 不超出最后一层
                if backbone.module_defs[next_idx]['type'] == 'convolutional':
                    next_conv = backbone.module_list[next_idx][0]
                    next_conv_weight = next_conv.weight.data
                    conv_sum = next_conv_weight.sum(dim=(2, 3))
                    offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                    if next_idx in CBL_idx:  # 对于convolutionnal，如果有BN，则该层卷积层不使用bias
                        next_bn = backbone.module_list[next_idx][1]
                        next_bn.running_mean.data.sub_(offset)
                        # next_conv.bias.data.add_(offset)
                    else:  # 如果无BN，则使用bias             针对mobile这样做，容易扩展
                        next_conv.bias.data.add_(offset)
                    # 补偿预测层
                    if backbone.module_defs[i]['feature'] == 'linear':  # 补偿预测层  （预测层不减枝，对于ssdlite可能可以改进)
                        i = int(backbone.module_defs[i]['feature_idx'])
                        # 补偿定位
                        reg_conv = predictor.reg_headers[i]
                        if isinstance(reg_conv, nn.Conv2d):
                            next_conv_weight = reg_conv.weight.data
                            # if cfg.QUANTIZATION.FINAL == True:
                            #     activation = quan_a(activation)
                            #     next_conv_weight = quan_w(next_conv_weight)
                            conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                            reg_conv.bias.data.add_(offset)
                        else:  # ssdlite深度可分离
                            next_conv_weight = reg_conv.conv[0].weight.data  # 深度卷积
                            # if cfg.QUANTIZATION.FINAL == True:
                            #     activation = quan_a(activation)
                            #     next_conv_weight = quan_w(next_conv_weight)
                            conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                            offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                            reg_conv.conv[1].running_mean.data.sub_(offset)

                        # 补偿分类
                        cls_conv = predictor.cls_headers[i]
                        if isinstance(cls_conv, nn.Conv2d):  # 普通ssdlite
                            next_conv_weight = cls_conv.weight.data
                            # if cfg.QUANTIZATION.FINAL == True:
                            #     activation = quan_a(activation)
                            #     next_conv_weight = quan_w(next_conv_weight)
                            conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                            cls_conv.bias.data.add_(offset)
                        else:  # ssdlite深度可分离
                            next_conv_weight = cls_conv.conv[0].weight.data
                            # if cfg.QUANTIZATION.FINAL == True:
                            #     activation = quan_a(activation)
                            #     next_conv_weight = quan_w(next_conv_weight)
                            conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                            offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                            cls_conv.conv[1].running_mean.data.sub_(offset)
            activations.append(activation)


    return pruned_model

def merge_mask(model, CBLidx2mask, CBLidx2filters):

    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)#第0维拼接
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()#并集
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())


def shortcut_prune(cfg,model,pruned_cfg,file,max,percent,quick,weight_path):
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])
    origin_nparameters = obtain_num_parameters(model)
    origin_size=model_size(model)
    #这里采用的shortcut方法，是https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone/blob/4516d76ba89b561983babd679543135484e7e9ac/slim_prune.py的方法
    CBL_idx, Conv_idx, prune_idx, _, _ = parse_module_defs(model.backbone.module_defs)

    # 将所有要剪枝的BN层的α参数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.backbone.module_list, prune_idx)
    # torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]
    thresh_index = int(len(bn_weights) * percent)
    thresh = sorted_bn[thresh_index].cuda()

    print(f'Global Threshold should be less than {thresh:.9f}.')

    predictor_channels = list(cfg.MODEL.BACKBONE.OUT_CHANNELS)
    # 获得保留的卷积核的个数和每层对应的mask,以及对应的head通道数
    num_filters, filters_mask, predictor_channels = obtain_filters_mask(model.backbone, thresh, CBL_idx, prune_idx,
                                                                        predictor_channels,  max)
    # CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}
    for i in model.backbone.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False
    print('merge the mask of layers connected to shortcut!')
    merge_mask(model.backbone, CBLidx2mask, CBLidx2filters)

    prune_and_eval(model, CBL_idx, CBLidx2mask,cfg)
    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()
    if quick ==0:
        print('实际剪枝---》')
        pruned_model = prune_model_keep_size(cfg,model, prune_idx, CBL_idx, CBLidx2mask)
        if max==0:
            with torch.no_grad():
                eval = do_evaluation(cfg,pruned_model, distributed=False)
            print('after prune_model_keep_size mAP is {}'.format(eval[0]['metrics']))       #对于最大剪枝，这里是不准的   相当于还没有截掉后面层

        # 获得原始模型的module_defs，并修改该defs中的卷积核数量
        compact_module_defs = deepcopy(model.backbone.module_defs)
        prune_after = -1  # cbl_idx索引号，后面的层都不要了（实际上看，是第一个BN偏置全零层最近的预测层之后的都被剪掉）  针对max
        if max == 1:
            new_predictor_channels = []
            for idx in CBL_idx:
                if model.backbone.module_defs[idx]['feature'] == 'linear' or model.backbone.module_defs[idx][
                    'feature'] == 'l2_norm':
                    i = int(model.backbone.module_defs[idx]['feature_idx'])
                    if predictor_channels[i] != -1:
                        new_predictor_channels.append(predictor_channels[i])
                        if i + 1 < len(predictor_channels):
                            if predictor_channels[i + 1] == -1:
                                prune_after = idx
                                break
                    if i + 1 == len(predictor_channels):
                        break
                elif model.backbone.module_defs[idx + 1]['type'] == 'shortcut' and model.backbone.module_defs[idx + 1][
                    'feature'] == 'linear':
                    i = int(model.backbone.module_defs[idx + 1]['feature_idx'])
                    new_predictor_channels.append(predictor_channels[i])  # 第一个short_cut连接head不会被裁掉
            predictor_channels = new_predictor_channels

        for idx, num in zip(CBL_idx, num_filters):
            assert compact_module_defs[idx]['type'] == 'convolutional'
            if idx==prune_after+1 and prune_after!=-1:
                compact_module_defs[idx]['filters']='-1'#这一层连同之后都不要
                break
            else:
                compact_module_defs[idx]['filters'] = str(num)

        write_cfg(pruned_cfg, compact_module_defs)
        print(f'Config file has been saved: {pruned_cfg}')
        cfg.MODEL.BACKBONE.OUT_CHANNELS=tuple(predictor_channels)
        print(f'PRUNED_MODEL.BACKBONE.OUT_CHANNELS:{cfg.MODEL.BACKBONE.OUT_CHANNELS}')
        cfg.MODEL.BACKBONE.CFG=pruned_cfg
        cfg.MODEL.BACKBONE.PRETRAINED=False   #定义模型时会加载预训练权重，这里不需要，因为之前的权重不匹配现在的通道数
        compact_model = build_detection_model(cfg)
        # print(compact_model)
        device = torch.device(cfg.MODEL.DEVICE)
        compact_model.to(device)
        init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask,prune_after)
        compact_nparameters = obtain_num_parameters(compact_model)
        compact_size = model_size(compact_model)
        random_input = torch.rand((16, 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)).to(device)
        pruned_forward_time = obtain_avg_forward_time(random_input, pruned_model)
        compact_forward_time = obtain_avg_forward_time(random_input, compact_model)
        # print(compact_model)
        # print(compact_model)
        with torch.no_grad():
            eval = do_evaluation(cfg, compact_model, distributed=False)
        print('Final pruned model mAP is {}'.format(eval[0]['metrics']))
        metric_table = [
            ["Metric", "Before", "After"],
            ["Parameters(M)", f"{origin_nparameters/(1024*1024)}", f"{compact_nparameters/(1024*1024)}"],
            ["模型体积(MB)", f"{origin_size}", f"{compact_size}"],
            ["Inference(ms)", f'{pruned_forward_time*1000/16:.4f}', f'{compact_forward_time*1000/16:.4f}']  #bs=16
        ]
        print(AsciiTable(metric_table).table)
        print(f'压缩率：{(origin_nparameters-compact_nparameters)/origin_nparameters}')
        file.write(f'PRUNED_MODEL.BACKBONE.OUT_CHANNELS:{cfg.MODEL.BACKBONE.OUT_CHANNELS}' + '\n')
        file.write(AsciiTable(metric_table).table + '\n')
        file.write(f'压缩率：{(origin_nparameters-compact_nparameters)/origin_nparameters}' + '\n')
        file.close()

        torch.save(compact_model.state_dict(),weight_path)
        print(f'Compact model has been saved.')