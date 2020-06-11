# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/18
from prune.normal_prune import normal_prune
from prune.sortcut_prune import shortcut_prune
import argparse
import torch
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
parser = argparse.ArgumentParser(description='PRUNE')
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--max",
    default=0,
    type=int,  # 是否采用最大剪枝，即如果有一层BN的权重全部小于阙值，认为后面的层都没有用，都应该不要    暂时无法对mobile使用，可对vgg使用，因为减少预测层后，head的构建模式导致深度可分离变为普通卷积，TODO
)
parser.add_argument(
    "--regular",
    help="是否规整剪枝(仅normal支持)",
    default=0,
    type=int,
)
parser.add_argument(
    "--percent",
    help="the percent of prune bn",
    default=0.05,
    type=float,
)
parser.add_argument(
    "--quick",
    help="是否仅快速看到剪枝效果",
    default=0,
    type=int,
)
parser.add_argument(
    "--model",
    default='model_final.pth',
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

if args.regular == 1:
    prune = 'regular_prune'
else:
    prune = 'prune'
if args.max == 1:
    prune = prune + '_max'
path = f'pruned_configs/{cfg.OUTPUT_DIR.split("/")[-1].split(".")[0]}_{cfg.PRUNE.TYPE}_{prune}_{args.percent}.txt'# 记录文件
file = open(path, 'w')
# 剪枝后的cfg文件
pruned_cfg = f'pruned_configs/{cfg.OUTPUT_DIR.split("/")[-1].split(".")[0]}_{cfg.PRUNE.TYPE}_{prune}_{args.percent}.cfg'
weights_path = f'pruned_model_weights/{cfg.OUTPUT_DIR.split("/")[-1].split(".")[0]}_{cfg.PRUNE.TYPE}_{prune}_{args.percent}.pth'
model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)
model.load_state_dict(torch.load(cfg.OUTPUT_DIR + '/' + str(args.model))['model'])

cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '/' + cfg.PRUNE.TYPE + '_' + prune + '_' + str(args.percent)

if cfg.PRUNE.TYPE=='normal':
    normal_prune(cfg,model,pruned_cfg,file,args.regular,args.max,args.percent,args.quick,weights_path)
elif cfg.PRUNE.TYPE=='shortcut':
    shortcut_prune(cfg,model,pruned_cfg,file,args.max,args.percent,args.quick,weights_path)

import os
os.system(f'rm -rf {cfg.OUTPUT_DIR}')  #删除产生的不必要中间文件

