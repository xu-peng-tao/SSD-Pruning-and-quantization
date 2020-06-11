##vgg with BN        dataset=oxfordhand
###train/resume     (训练中evaluate是单gpu)
```
one gpu:
CUDA_VISIBLE_DEVICES="2" python train.py --config-file configs/vgg_bn_ssd300_hand.yaml
tow gpu：
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/vgg_bn_ssd300_hand.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```
###evaluate         (只支持单gpu)
```
CUDA_VISIBLE_DEVICES="2" python test.py --config-file configs/vgg_bn_ssd300_hand.yaml
```
###demo
```
CUDA_VISIBLE_DEVICES="2" python demo.py --config-file configs/vgg_bn_ssd300_hand.yaml --ckpt /path_to/model_002500.pth --dataset_type oxfordhand --score_threshold 0.4
```

##mobile_netv2        dataset=voc
###train/resume        
```
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/mobile_v2_ssd_voc0712.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```
mAP:model_final-->70.66


##mobile_netv2        dataset=hand          prune=normal_sparse   (稀疏化需要轮数多一些训练，才会容易剪枝)
###train/resume        
```
one gpu:
CUDA_VISIBLE_DEVICES="3" python train.py --config-file configs/mobile_v2_ssd_hand_normal_sparse.yaml
two:
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/mobile_v2_ssd_hand_normal_sparse.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```
###normal prune
```
CUDA_VISIBLE_DEVICES="3" python prune.py --config-file configs/mobile_v2_ssd_hand_normal_sparse.yaml --regular 0 --percent 0.1 --quick 0 --model model_final.pth
```


##mobile_netv2        dataset=hand          prune=shortcut_sparse
###train/resume        
```
one gpu:
CUDA_VISIBLE_DEVICES="2" python train.py --config-file configs/mobile_v2_ssd_hand_shortcut_sparse.yaml
two:
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/mobile_v2_ssd_hand_shortcut_sparse.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```
###shortcut prune
```
CUDA_VISIBLE_DEVICES="3" python prune.py --config-file configs/mobile_v2_ssd_hand_shortcut_sparse.yaml --percent 0.2 --quick 0 --model model_final.pth
```
