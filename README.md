#SSD-Pruning and quantification

1，在SSD上实现模型压缩：剪枝和量化

2，模型压缩支持多backbone（目前包括mobile-netv2-SSD、vgg16-BN-SSD),并容易扩展到其他backbone

3，SSD源码来自于[lufficc/SSD](https://github.com/lufficc/SSD),剪枝方法参考[SpursLipu /YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining-Multibackbone),
量化方法参考[666DZY666/model-compression](https://github.com/666DZY666/model-compression) ,在此致谢。

##Dataset
###COCO
Microsoft COCO: Common Objects in Context
####Download COCO 2014
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
原始数据集可由[官网](http://www.robots.ox.ac.uk/~vgg/data/hands)下载,本项目将数据集格式进行转化。转换格式的数据集可在[百度网盘]()下载。下载解压后得到images和labels两个文件夹，然后configs/oxfordhand.data中的对应路径更换成解压后文件的路径即可。
###For more dataset
可以将新的数据集变为oxford hand数据集格式，建立对应的.names和.data文件即可。


##Backbone
原始[lufficc/SSD](https://github.com/lufficc/SSD)中有多backbone的实现，本代码依然兼容。这里为便于扩展新的backbone和便于模型剪枝，采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)中cfg文件的形式定义backbone。目前支持vgg16-BN、vgg-BN-fpga、mobilenet_v2,可以添加新的cfg文件以支持更多的backbone(若有新的结构，需要在ssd/modeling/backbone/backbone_cfg.py中进行添加定义)。当定义了新的cfg文件，可定义对应的yaml文件，使用test_model_structure.py打印模型结构、使用get_model_size得到模型规模。


##Train
```
one gpu:
CUDA_VISIBLE_DEVICES="2" python train.py --config-file configs/*.yaml
tow gpu：
export NGPUS=2
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/*.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000 
```

##Evaluate 
```
CUDA_VISIBLE_DEVICES="2" python test.py --config-file configs/*.yaml
```
##Demo
```
CUDA_VISIBLE_DEVICES="2" python demo.py --config-file configs/*.yaml --ckpt /path_to/*.pth --dataset_type oxfordhand --score_threshold 0.4
```

##Prune
剪枝方法来源于论文[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)，剪枝无需微调方法来源于[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124?context=cs)。

剪枝分为3步：

1，在yaml文件中定义PRUNE的TYPE和SR：TYPE为剪枝的类型，可以从'normal'和'shortcut'中选择，'normal'为正常剪枝（不对shortcut进行剪枝），'shortcut'为极限剪枝（对shortcut进行剪枝，剪枝率高）。SR为稀疏因子大小。如configs/mobile_v2_ssd_hand_normal_sparse.yaml和configs/mobile_v2_ssd_hand_shortcut_sparse.yaml所示。

2，进行稀疏化训练：
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

##Quantification
TODO......

##Experiment
部分实验训练、测试等具体命令可见[experiment.md](experiment.md)。部分实验结果可见[result.md](result.md)。








