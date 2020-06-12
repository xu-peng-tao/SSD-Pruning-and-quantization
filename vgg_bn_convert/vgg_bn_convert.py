# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/13
#convert vgg_16_bn.pth to vgg_16_bn_reducefc_pth
#wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
#https://blog.csdn.net/lyl771857509/article/details/84175874  VGG16、VGG16_bn、VGG19_bn详解以及使用pytorch进行模型预训练
#https://github.com/pytorch/vision/tree/master/torchvision/models
import torch
from ssd.config import cfg
from ssd.modeling.backbone import vgg
import torchvision.models as models
import os
#convert
vgg_model = models.vgg16_bn()
state=torch.load('vgg16_bn-6c64b313.pth')
# for k,v in state.items():
#     print(k,v.size())
vgg_model.load_state_dict(state)
# print(vgg_model.features)  #reduce_fc
torch.save(vgg_model.features.state_dict(),'vgg16_bn_reducefc.pth')


#test1
# state=torch.load('vgg16_bn_reducefc.pth')
# # print(state)
# cfg.MODEL.BACKBONE.WITHBN=True
# model=vgg.VGG(cfg)
# print(model.vgg[:42])
# model.vgg[:42].load_state_dict(state) #加载前面层



#test2
# import numpy as np
# with open('vgg16_bn_reducefc.pth', 'rb') as f:
#     weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
# print(weights.shape)


#test3
# from ssd.modeling.backbone import Backbone
# state=torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
# for i,name in enumerate(state):
#     print(str(i)+':')
#     print(name)
# model=Backbone(cfg)
# print(model)
# if cfg.MODEL.BACKBONE.WEIGHTS.find('ssd')>=0:
#     print('暂时')
# elif cfg.MODEL.BACKBONE.WEIGHTS.find('vgg16_bn_reduce')>=0:
#     name_list=list(state.keys())
#     num=0
#     # print(state[name_list[0]])
#     cut_off=17  #vgg16_bn_reduce
#     for i, (mdef, module) in enumerate(zip(model.module_defs[:cut_off], model.module_list[:cut_off])):
#         if mdef['type'] == 'convolutional':
#             conv_layer = module[0]
#             conv_layer.weight.data.copy_(state[name_list[num]])
#             num=num+1
#             conv_layer.bias.data.copy_(state[name_list[num]])
#             num = num + 1
#             if mdef['batch_normalize'] == '1':
#                 # Load BN bias, weights, running mean and running variance
#                 bn_layer = module[1]
#                 bn_layer.weight.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.bias.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.running_mean.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.running_var.data.copy_(state[name_list[num]])
#                 num = num + 2 #跳过num_batches_tracked


#test4
# from ssd.modeling.backbone import Backbone
# config_file='../configs/vgg_ssd300_hand_fpga.yaml'
# cfg.merge_from_file(config_file)
# cfg.freeze()
# state=torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
# for i,name in enumerate(state):
#     print(str(i)+':')
#     print(name)
# model=Backbone(cfg)
# print(model)
# if cfg.MODEL.BACKBONE.WEIGHTS.find('ssd')>=0:
#     print('暂时')
# elif cfg.MODEL.BACKBONE.WEIGHTS.find('vgg16')>=0:
#     # print('1')
#     name_list=list(state.keys())
#     num=0
#     # print(state[name_list[0]])
#     cut_off=17  #vgg_bn:vgg16_bn_reduce       vgg_fpga:没到空洞
#     for i, (mdef, module) in enumerate(zip(model.module_defs[:cut_off], model.module_list[:cut_off])):
#         if mdef['type'] == 'convolutional':
#             conv_layer = module[0]
#             conv_layer.weight.data.copy_(state[name_list[num]])
#             num=num+1
#             conv_layer.bias.data.copy_(state[name_list[num]])
#             num = num + 1
#             if mdef['batch_normalize'] == '1':
#                 # Load BN bias, weights, running mean and running variance
#                 bn_layer = module[1]
#                 bn_layer.weight.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.bias.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.running_mean.data.copy_(state[name_list[num]])
#                 num = num + 1
#                 bn_layer.running_var.data.copy_(state[name_list[num]])
#                 num = num + 2 #跳过num_batches_tracked

#test5
os.environ['CUDA_VISIBLE_DEVICES']="1"
from ssd.modeling.backbone import Backbone
config_file='../configs/vgg_ssd300_hand_fpga.yaml'
cfg.merge_from_file(config_file)
cfg.freeze()
state=torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
for i,name in enumerate(state):
    print(str(i)+':')
    print(name)
model=Backbone(cfg)
print(model)
if cfg.MODEL.BACKBONE.WEIGHTS.find('ssd')>=0:
    state=state['model']
    name_list = list(state.keys())
    num = 0
    for i, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
        print(mdef)
        if mdef['type'] == 'convolutional':
            # print('1')
            conv_layer = module[0]
            conv_layer.weight.data.copy_(state[name_list[num]])
            num = num + 1
            conv_layer.bias.data.copy_(state[name_list[num]])
            num = num + 1
            if mdef['batch_normalize'] == '1':
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                bn_layer.weight.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.bias.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.running_mean.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.running_var.data.copy_(state[name_list[num]])
                num = num + 2  # 跳过num_batches_tracked

elif cfg.MODEL.BACKBONE.WEIGHTS.find('vgg16')>=0:
    # print('1')
    name_list=list(state.keys())
    num=0
    # print(state[name_list[0]])
    cut_off=17  #vgg_bn:vgg16_bn_reduce       vgg_fpga:没到空洞
    for i, (mdef, module) in enumerate(zip(model.module_defs[:cut_off], model.module_list[:cut_off])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            conv_layer.weight.data.copy_(state[name_list[num]])
            num=num+1
            conv_layer.bias.data.copy_(state[name_list[num]])
            num = num + 1
            if mdef['batch_normalize'] == '1':
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                bn_layer.weight.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.bias.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.running_mean.data.copy_(state[name_list[num]])
                num = num + 1
                bn_layer.running_var.data.copy_(state[name_list[num]])
                num = num + 2 #跳过num_batches_tracked


