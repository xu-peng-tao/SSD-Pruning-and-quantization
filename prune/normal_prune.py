# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/16
import torch.nn as nn
from copy import deepcopy
from ssd.modeling.detector import build_detection_model
from ssd.engine.inference import do_evaluation
from prune.prune_utils import *
from get_model_size import model_size
from terminaltables import AsciiTable
filter_switch = [8, 16, 32, 64, 128, 256, 512, 1024]  #规整通道数
def parse_module_defs(module_defs):
    CBL_idx = []
    Conv_idx = []#普通卷积和depthwise
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
        elif module_def['type'] == 'depthwise':
            Conv_idx.append(i)
        # l2_norm前的不剪枝  原始vgg_ssd
        if module_def['feature'] == 'l2_norm':
            ignore_idx.add(i)
    for i, module_def in enumerate(module_defs):
        # 跳连层的前一层不剪,跳连层的来源层不剪
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
        # 深度可分离卷积层的其前一层不剪  待改进
        if module_def['type'] == 'depthwise':
            ignore_idx.add(i - 1)
        # 上采样层前的卷积层不裁剪
        if module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)
        #maxpool
        if module_def['type'] == 'maxpool':
            # 影响不用微调补偿：经最大池化影响(暂未考虑平均池化）   一个特征图上所有点一样去补偿，有的补零padding，最大会变（bn.bias为负），补偿下一层不准
            # relu能保证一定为非负
            if module_defs[i-1]['activation'] == 'relu':
                pass
            else:
                if int(module_def[i ]['pad']) == 0:  # maxpool不补0
                    pass
                else: # 其他情况maxpool之前的不减枝
                    ignore_idx.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx

# 该函数有很重要的意义：
# ①先用深拷贝将原始模型拷贝下来，得到model_copy
# ②将model_copy中，BN层中低于阈值的α参数赋值为0
# ③在BN层中，输出y=α*x+β，由于α参数的值被赋值为0，因此输入仅加了一个偏置β
# ④很神奇的是，network slimming中是将α参数和β参数都置0，该处只将α参数置0，但效果却很好：其实在另外一篇论文中，已经提到，可以先将β参数的效果移到
# 下一层卷积层，再去剪掉本层的α参数
# 该函数用最简单的方法，让我们看到了，如何快速看到剪枝后的效果
#未分规整的，规整的应该比这个好一些，多一点通道
def prune_and_eval(model, sorted_bn, prune_idx,percent,cfg):       #不是最终准确的结果
    print(f'mAP of the original model is:')
    with torch.no_grad():
        eval = do_evaluation(cfg, model, distributed=False)
        print(eval[0]['metrics'])
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * percent)
    # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    thre = sorted_bn[thre_index]
    thre = thre.cuda()

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')
    remain_num = 0
    for idx in prune_idx:
        bn_module = model_copy.backbone.module_list[idx][1]
        mask = bn_module.weight.data.abs().ge(thre).float()
        remain_num += int(mask.sum())
        bn_module.weight.data.mul_(mask)
    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1 - remain_num / len(sorted_bn):.3f}')
    print('快速看剪枝效果----》')
    print(f'mAP of the pruned model is:')
    with torch.no_grad():
        eval=do_evaluation(cfg, model_copy, distributed=False)
        print(eval[0]['metrics'])
    return thre

def obtain_filters_mask(model, thre, CBL_idx, prune_idx,predictor_channels,regular,max):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    prune_=0#为1时表示找到第一个BN偏置全零层   max时后面的层都不要
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:

            mask = bn_module.weight.data.abs().ge(thre).float().cpu().numpy()
            mask_cnt = int(mask.sum())
            if max==1:#最大剪枝
                if mask_cnt==0:
                    prune_=1


            if regular==1:#规整剪枝
                if mask_cnt == 0:
                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    sort_bn_values = torch.sort(this_layer_sort_bn)[0]
                    bn_cnt = bn_module.weight.shape[0]
                    this_layer_thre = sort_bn_values[bn_cnt - 8]
                else:
                    for i in range(len(filter_switch)):
                        if mask_cnt <= filter_switch[i]:
                            mask_cnt = filter_switch[i]
                            break
                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    sort_bn_values = torch.sort(this_layer_sort_bn)[0]
                    bn_cnt = bn_module.weight.shape[0]
                    this_layer_thre = sort_bn_values[bn_cnt - mask_cnt]
                mask = bn_module.weight.data.abs().ge(this_layer_thre.cuda()).float().cpu().numpy()
                remain = int(mask.sum())
            else:
                remain=mask_cnt
            if remain == 0:
                # print("Channels would be all pruned!") #当有一层被完全剪掉，yolo采用了阙值，不让这么剪，我这里可以保留这个一个最大的，其余剪掉
                # raise Exception
                remain=1
                max=bn_module.weight.data.abs().max()
                mask = bn_module.weight.data.abs().ge(max).float().cpu().numpy()
            pruned = pruned + mask.shape[0] - remain



            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.copy())
        if model.module_defs[idx]['feature'] == 'linear':
            i = int(model.module_defs[idx]['feature_idx'])
            if prune_==1:
                predictor_channels[i] =-1  #不要这个预测层了
            else:
                predictor_channels[i] = remain  # 预测层输入通道数改变
                # print(f'remian{remain}')
    #因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数   除去了l2层，也进行了防止一层被裁掉的操作
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')     #对于最大剪枝，这里是不准的

    return num_filters, filters_mask,predictor_channels

def prune_model_keep_size(cfg,model, prune_idx, CBL_idx, CBLidx2mask,Conv_idx):
#beta参数补偿到后面卷积层
    pruned_model = deepcopy(model)
    backbone=pruned_model.backbone
    predictor=pruned_model.box_head.predictor
    # if cfg.QUANTIZATION.TYPE== 'dorefa':
    #     quan_w=dorefa.weight_quantize(w_bits=cfg.QUANTIZATION.WBITS)
    #     quan_a=dorefa.activation_quantize(a_bits=cfg.QUANTIZATION.ABITS)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = backbone.module_list[idx][1]
        bn_module.weight.data.mul_(mask)  #bn w 剪掉
        activate = backbone.module_list[idx][-1]  #有可能cb没有l  这里就是bn
        bn_bias = bn_module.bias.data
        next_idx=idx+1  #补偿下一层
        if next_idx != len(backbone.module_defs):  # 不超出最后一层
            if next_idx not in CBL_idx:
                if next_idx not in Conv_idx:  #下一层是池化
                    next_idx=next_idx+1
            next_conv = backbone.module_list[next_idx][0]
            next_conv_weight = next_conv.weight.data
            if isinstance(activate,nn.BatchNorm2d):
                activation=(1 - mask) * bn_bias
            else:
                activation = activate((1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》未剪掉的为0，得保证0量化为0
            # if backbone.module_defs[next_idx]['quantization'] == '1':
            #     activation=quan_a(activation)
            #     next_conv_weight=quan_w(next_conv_weight)
            # 池化影响在求prune_idx时已解决
            conv_sum=next_conv_weight.sum(dim=(2,3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
            if next_idx in CBL_idx:#对于convolutionnal，如果有BN，则该层卷积层不使用bias
                next_bn = backbone.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
                #next_conv.bias.data.add_(offset)
            else: #如果无BN，则使用bias             针对mobile这样做，容易扩展
                next_conv.bias.data.add_(offset)


        #补偿预测层
        if backbone.module_defs[idx]['feature'] == 'linear':# 补偿预测层  （预测层不减枝，对于ssdlite可能可以改进)
            i=int(backbone.module_defs[idx]['feature_idx'])
            #补偿定位
            reg_conv=predictor.reg_headers[i]
            if isinstance(activate, nn.BatchNorm2d):
                activation = (1 - mask) * bn_bias
            else:
                activation = activate((1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》为剪掉的为0，得保证0量化为0
            if isinstance(reg_conv,nn.Conv2d):
                next_conv_weight = reg_conv.weight.data
                # if cfg.QUANTIZATION.FINAL == True:
                #     activation = quan_a(activation)
                #     next_conv_weight = quan_w(next_conv_weight)
                conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                reg_conv.bias.data.add_(offset)
            else:#ssdlite深度可分离
                next_conv_weight = reg_conv.conv[0].weight.data  # 深度卷积
                # if cfg.QUANTIZATION.FINAL == True:
                #     activation = quan_a(activation)
                #     next_conv_weight = quan_w(next_conv_weight)
                conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                reg_conv.conv[1].running_mean.data.sub_(offset)

            #补偿分类
            cls_conv = predictor.cls_headers[i]
            if isinstance(activate, nn.BatchNorm2d):
                activation = (1 - mask) * bn_bias
            else:
                activation = activate((1 - mask) * bn_bias)  # 先过激活，再过量化（如果有），（1-mask)--->被剪掉的为1---》为剪掉的为0，得保证0量化为0
            if isinstance(cls_conv,nn.Conv2d):#普通ssdlite
                next_conv_weight = cls_conv.weight.data
                # if cfg.QUANTIZATION.FINAL == True:
                #     activation = quan_a(activation)
                #     next_conv_weight = quan_w(next_conv_weight)
                conv_sum = next_conv_weight.sum(dim=(2, 3))  # outchannel,inchannel
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # outchannel
                cls_conv.bias.data.add_(offset)
            else:# ssdlite深度可分离
                next_conv_weight = cls_conv.conv[0].weight.data
                # if cfg.QUANTIZATION.FINAL == True:
                #     activation = quan_a(activation)
                #     next_conv_weight = quan_w(next_conv_weight)
                conv_sum = next_conv_weight.sum(dim=(2, 3)).reshape(-1)  # outchannel
                offset = conv_sum.matmul(activation.reshape(-1)).reshape(-1)  # outchannel
                cls_conv.conv[1].running_mean.data.sub_(offset)


        bn_module.bias.data.mul_(mask)    #bn bias 剪掉

    return pruned_model

def normal_prune(cfg,model,pruned_cfg,file,regular,max,percent,quick,weight_path):
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])
    origin_nparameters = obtain_num_parameters(model)
    origin_size=model_size(model)
    CBL_idx, Conv_idx, prune_idx=parse_module_defs(model.backbone.module_defs)

    # 将所有要剪枝的BN层的α参数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.backbone.module_list, prune_idx)
    # torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    # highest_thre = []
    # for idx in prune_idx:
    #     # .item()可以得到张量里的元素值
    #     highest_thre.append(model.backbone.module_list[idx][1].weight.data.abs().max().item())
    # highest_thre = min(highest_thre)
    # 找到highest_thre对应的下标对应的百分比
    # percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

    # print(f'Threshold should be less than {highest_thre:.4f}.')
    # print(f'The corresponding prune ratio should less than {percent_limit:.3f}.')      #这一行的限制只是为了防止normal剪枝某一层减空，如果保留这一层的一个，则没有这个限制了

    thre=prune_and_eval(model,sorted_bn,prune_idx,percent,cfg)
    if quick ==0:
        print('实际剪枝---》')
        predictor_channels=list(cfg.MODEL.BACKBONE.OUT_CHANNELS)
        # 获得保留的卷积核的个数和每层对应的mask,以及对应的head通道数
        num_filters, filters_mask,predictor_channels=obtain_filters_mask(model.backbone, thre, CBL_idx, prune_idx,predictor_channels,regular,max)
        # CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
        CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
        pruned_model = prune_model_keep_size(cfg,model, prune_idx, CBL_idx, CBLidx2mask,Conv_idx)
        if max==0:
            with torch.no_grad():
                eval = do_evaluation(cfg,pruned_model, distributed=False)
            print('after prune_model_keep_size mAP is {}'.format(eval[0]['metrics']))       #对于最大剪枝，这里是不准的   相当于还没有截掉后面层

        # 获得原始模型的module_defs，并修改该defs中的卷积核数量
        compact_module_defs = deepcopy(model.backbone.module_defs)
        prune_after = -1  # cbl_idx索引号，后面的层都不要了（实际上看，是第一个BN偏置全零层最近的预测层之后的都被剪掉）  针对max
        if max==1:
            new_predictor_channels=[]
            for idx in CBL_idx:
                if model.backbone.module_defs[idx]['feature'] == 'linear' or model.backbone.module_defs[idx]['feature'] =='l2_norm':
                    i = int(model.backbone.module_defs[idx]['feature_idx'])
                    if predictor_channels[i] != -1:
                        new_predictor_channels.append(predictor_channels[i])
                        if i + 1 < len(predictor_channels):
                            if predictor_channels[i + 1] == -1:
                                prune_after = idx
                                break
                    if i+1==len(predictor_channels):
                        break
                elif model.backbone.module_defs[idx+1]['type']=='shortcut' and model.backbone.module_defs[idx+1]['feature']=='linear':
                    i = int(model.backbone.module_defs[idx+1]['feature_idx'])
                    new_predictor_channels.append(predictor_channels[i])#第一个short_cut连接head不会被裁掉
            predictor_channels=new_predictor_channels

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