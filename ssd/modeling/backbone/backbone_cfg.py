# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/14
from ssd.layers import L2Norm
import torch.nn.functional as F
import torch.nn as nn
import torch
from ssd.modeling import registry
import os
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class relu6(nn.Module):
    def __init__(self):
        super(relu6, self).__init__()

    def forward(self, x):
        return F.relu6(x, inplace=True)


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()

    def forward(self, x):
        return x * (F.relu6(x + 3.0, inplace=True) / 6.0)


class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        out = F.relu6(x + 3.0, inplace=True) / 6.0
        return out


class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            h_sigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.module_defs = self.parse_model_cfg(cfg.MODEL.BACKBONE.CFG)
        self.module_list, self.l2_norm_index,self.features,self.l2_norm,self.routs= self.create_backbone(cfg,self.module_defs)
        self.reset_parameters()



    def parse_model_cfg(self,path):
        # Parses the ssd layer configuration file and returns module definitions
        # print(os.getcwd())#绝对路径
        # print(os.path.abspath(os.path.join(os.getcwd(), "../../..")))  #获取上上上级目录
        # file = open(os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../../..")),path), 'r')  #测试本文件时使用
        file=open(path,'r')
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        mdefs = []  # module definitions
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                mdefs.append({})
                mdefs[-1]['type'] = line[1:-1].rstrip()
                if mdefs[-1]['type'] == 'convolutional' or mdefs[-1]['type'] == 'depthwise':
                    mdefs[-1]['batch_normalize'] = '0'  # pre-populate with zeros (may be overwritten later)
                    mdefs[-1]['feature'] = 'no'
                    mdefs[-1]['dilation'] = '1'
                    mdefs[-1]['bias'] = '1'
                else:
                    mdefs[-1]['feature'] = 'no'
            else:
                key, val = line.split("=")
                key = key.rstrip()
                mdefs[-1][key] = val.strip()
        return mdefs

    def create_backbone(self,cfg, module_defs):
        # Constructs module list of layer blocks from module configuration in module_defs
        output_filters = [int(cfg.INPUT.CHANNELS)]
        module_list = nn.ModuleList()
        features = []  # list of layers which rout to detection layes
        routs = []  # list of layers which rout to deeper layes
        l2_norm_index = 0
        l2_norm = 0
        for i, mdef in enumerate(module_defs):
            modules = nn.Sequential()
            # print(mdef)
            if mdef['type'] == 'convolutional':
                bn = int(mdef['batch_normalize'])
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                pad = int(mdef['pad'])
                if mdef['bias']=='1':
                    bias=True
                else:
                    bias=False

                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=kernel_size,
                                                       stride=int(mdef['stride']),
                                                       padding=pad,
                                                       dilation=int(mdef['dilation']),bias=bias))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))


                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                elif mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU(inplace=True))
                elif mdef['activation'] == 'relu6':
                    modules.add_module('activation', relu6())
                elif mdef['activation'] == 'h_swish':
                    modules.add_module('activation', h_swish())
            elif mdef['type'] == 'depthwise':
                bn = int(mdef['batch_normalize'])
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                pad = int(mdef['pad'])
                if mdef['bias'] == '1':
                    bias = True
                else:
                    bias = False

                modules.add_module('DepthWise2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=kernel_size,
                                                       stride=int(mdef['stride']),
                                                       padding=pad,
                                                       groups=output_filters[-1],
                                                       dilation=int(mdef['dilation']), bias=bias))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                elif mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU(inplace=True))
                elif mdef['activation'] == 'relu6':
                    modules.add_module('activation', relu6())
                elif mdef['activation'] == 'h_swish':
                    modules.add_module('activation', h_swish())


            elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
                layer = int(mdef['from'])
                filters = output_filters[layer]
                routs.extend([i + layer if layer < 0 else layer])

            elif mdef['type'] == 'maxpool':
                kernel_size = int(mdef['size'])
                stride = int(mdef['stride'])
                ceil_mode = True if int(mdef['ceil_mode']) else False
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int(mdef['pad']),
                                       ceil_mode=ceil_mode)#https://www.cnblogs.com/xxxxxxxxx/p/11529343.html ceilmode
                modules = maxpool
            else:
                print("Error type!")
                raise Exception
            if mdef['feature'] == 'linear':  # 传入预测层
                features.append(i)
            elif mdef['feature'] == 'l2_norm':
                features.append(i)
                l2_norm_index = i
                l2_norm = L2Norm(filters)
            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return module_list, l2_norm_index, features, l2_norm,routs
    def reset_parameters(self):
        for m in self.module_list.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        layer_outputs = []
        features=[]
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'depthwise', 'se', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            layer_outputs.append(x if i in self.routs else [])
            if i in self.features:
                if i!=self.l2_norm_index:
                    features.append(x)
                else:#l2_norm
                    feature=self.l2_norm(x)
                    features.append(feature)
        return tuple(features)
@registry.BACKBONES.register('cfg_backbone')
def cfg_backbone(cfg, pretrained=True):
    model = Backbone(cfg)
    if pretrained:
        # print('Load pretrained model from {}'.format(cfg.MODEL.BACKBONE.WEIGHTS))
        state = torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
        if cfg.MODEL.BACKBONE.WEIGHTS.find('ssd') >= 0:  #在检测数据集(voc\coco)上的预训练模型，加载全部backbone
            if 'model' in state.keys():
                state = state['model']
            name_list = list(state.keys())
            num = 0
            for i, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
                if mdef['type'] == 'convolutional' or mdef['type'] == 'depthwise':
                    conv_layer = module[0]
                    conv_layer.weight.data.copy_(state[name_list[num]])
                    num = num + 1
                    if conv_layer.bias is not None:
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
        else:#加载在分类数据集（imagenet)上的权重，只加载reduce_fc的部分
            name_list = list(state.keys())
            num = 0
            for i, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
                if mdef['type'] == 'convolutional' or mdef['type'] == 'depthwise':
                    conv_layer = module[0]
                    conv_layer.weight.data.copy_(state[name_list[num]])
                    num = num + 1
                    if conv_layer.bias is not None:
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
                if 'backbone' in mdef.keys():#加载在分类数据集（imagenet)上的权重，只加载reduce_fc的部分,之后不加载
                    break
    return model
if __name__=='__main__':
    from ssd.config import cfg
    model = Backbone(cfg)
    print(model)


