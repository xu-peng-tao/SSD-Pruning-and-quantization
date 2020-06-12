# SSD-Pruning and quantization

1，在SSD上实现模型压缩：剪枝和量化

2，模型压缩支持多backbone（目前包括mobile-netv2-SSD、vgg16-BN-SSD),并容易扩展到其他backbone

3，SSD源码来自于[lufficc/SSD](https://github.com/lufficc/SSD),剪枝方法参考[SpursLipu /YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone),
量化方法参考[666DZY666/model-compression](https://github.com/666DZY666/model-compression) ,在此致谢。

4，项目环境：Python 3.6.10 ;Torch 1.4.0。其他环境配置参考：[lufficc/SSD](https://github.com/lufficc/SSD)

## Dataset
### COCO
Microsoft COCO: Common Objects in Context
#### Download COCO 2014
```Shell
sh ssd/data/datasets/scripts/COCO2014.sh
```
### VOC Dataset
PASCAL VOC: Visual Object Classes
#### Download VOC2007 trainval & test
```Shell
sh ssd/data/datasets/scripts/VOC2007.sh
```
#### Download VOC2012 trainval
```Shell
sh ssd/data/datasets/scripts/VOC2012.sh
```
### oxford hand
原始数据集可由[官网](http://www.robots.ox.ac.uk/~vgg/data/hands)下载,本项目将数据集格式进行转化。转换格式的数据集可在[百度网盘](https://pan.baidu.com/s/1n2KG6Y7DrlhqdVxRY4nOhg)(提取码：w4av))下载。下载解压后得到images和labels两个文件夹，然后configs/oxfordhand.data中的对应路径更换成解压后文件的路径即可。
### For more dataset
可以将新的数据集变为oxford hand数据集格式，建立对应的.names和.data文件即可。


## Backbone
原始[lufficc/SSD](https://github.com/lufficc/SSD)中有多backbone的实现，本代码依然兼容。这里为便于扩展新的backbone和便于模型剪枝，采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)中cfg文件的形式定义backbone。目前支持vgg16-BN、vgg-BN-fpga、mobilenet_v2,可以添加新的cfg文件以支持更多的backbone(若有新的结构，需要在ssd/modeling/backbone/backbone_cfg.py中进行添加定义)。当定义了新的cfg文件，可定义对应的yaml文件，使用test_model_structure.py打印模型结构、使用get_model_size得到模型规模。


## Train
```
one gpu:
CUDA_VISIBLE_DEVICES="2" python train.py --config-file configs/*.yaml
tow gpu：
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/*.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```

## Evaluate 
TEST.BN_FUSE True 表示测试时对BN进行融合。
```
CUDA_VISIBLE_DEVICES="2" python test.py --config-file configs/*.yaml TEST.BN_FUSE True
```
## Demo
```
CUDA_VISIBLE_DEVICES="2" python demo.py --config-file configs/*.yaml --ckpt /path_to/*.pth --dataset_type oxfordhand --score_threshold 0.4 TEST.BN_FUSE True
```

## Prune
剪枝方法来源于论文[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)，剪枝无需微调方法来源于[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124?context=cs)。

剪枝分为3步：

1，在yaml文件中定义PRUNE的TYPE和SR：TYPE为剪枝的类型，可以从'normal'和'shortcut'中选择，'normal'为正常剪枝（不对shortcut进行剪枝），'shortcut'为极限剪枝（对shortcut进行剪枝，剪枝率高）。SR为稀疏因子大小。如configs/mobile_v2_ssd_hand_normal_sparse.yaml和configs/mobile_v2_ssd_hand_shortcut_sparse.yaml所示。

2，进行稀疏化训练：

可以从头开始训练，也可以从之前非稀疏化的权重开始训练:在yaml文件中设置MODEL.FINE_TUNE和MODEL.WEIGHTS。
```
one gpu:
CUDA_VISIBLE_DEVICES="3" python train.py --config-file configs/*_sparse.yaml
two:
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/*_sparse.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```

3,进行剪枝：

--percent 为剪枝率，若进行较高剪枝率的剪枝，需要做好稀疏化训练（训练轮数较长，稀疏因子设置得当）

--regular 取0或1，0表示不进行规整剪枝，1表示进行规整剪枝。规整剪枝使剪枝后的通道数均为8的倍数，主要用于硬件部署。只对normal剪枝设置有效。

--max 取0或1。有时针对某个特定数据集，backbone并不需要输出全部的分支到predict head，当有一层BN的权重全部小于阙值，认为后面的层都没有用。当取1时，会把后面的层剪掉。

```
CUDA_VISIBLE_DEVICES="3" python prune.py --config-file configs/*_sparse.yaml --regular 0 --max 0 --percent 0.1 --model model_final.pth
```
注:本项目提供的剪枝策略，从理论上不需要进行剪枝后微调。但经实验，若采用较大的剪枝率，mAP掉的很多的情况下，微调仍会起到很重要的作用。
## Quantization
参考论文：

[BinarizedNeuralNetworks: TrainingNeuralNetworkswithWeightsand ActivationsConstrainedto +1 or−1](https://arxiv.org/abs/1602.02830)

[XNOR-Net:ImageNetClassiﬁcationUsingBinary ConvolutionalNeuralNetworks](https://arxiv.org/abs/1603.05279)

[Ternary weight networks](https://arxiv.org/abs/1605.04711)

[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)

[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

[Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)

量化分为3步：

1，新建量化backbone网络cfg文件，将需要量化的层quantization=1，如configs/vgg_bn_ssd300_fpga_quan.cfg。

2，在yaml文件中定义QUANTIZATION的TYPE、FINAL、WBITS、ABITS：TYPE为量化的类型，可以从'dorefa'、'IAO'、'BWN'中选择。FINAL表示predict head是否量化。WBITS、ABITS为量化的位数（dorefa\IAO)或量化为几值（BWN，BWN支持权重二/三值 、激活二值)。如configs/vgg_bn_ssd300_hand_fpga_sparse_quan_w8a8.yaml等。

3，进行量化训练：
```
one gpu:
CUDA_VISIBLE_DEVICES="3" python train.py --config-file configs/*.yaml
two:
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/*.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```
## Pruning and quantization

先量化后剪枝：量化训练时同时进行稀疏化训练，如configs/vgg_bn_ssd300_hand_fpga_sparse_quan_w8a8.yaml。然后直接进行剪枝。量化后剪枝目前只支持dorefa匹配normal方法。

先剪枝后量化：剪枝完后得到剪枝后的网络cfg文件、txt文件（在pruned_configs文件夹下）和权重文件（在pruned_model_weights文件夹下)，根据它们定义yaml文件进行量化训练。

## Get weights
可使用get_weights.py和get_weights_bin.py得到模型参数用于模型部署。

## Experiment
部分实验训练、测试等具体命令可见[experiment.md](experiment.md)。部分实验结果可见[result.md](result.md)。








