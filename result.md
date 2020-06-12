# Pruning and quantization
下面的实验均在oxfordhand数据集上进行;Pruning后均未微调。



## Pruning

0表示层数不变，-3表示少了后三层。            
#### vgg_bn_ssd300
| 模型 | 参数量 |模型体积  |压缩率  |耗时  |mAP  |层数  |
| --- | --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |7.48ms  |0.7815  |   0|
| normal剪枝| 4.4M |17.6MB  |80.6%  |5.64ms  |0.7700  |  0|
| normal剪枝(max)| 4.4M |17.6MB  |80.6%  |5.58ms  |0.7700  |  -3|
| normal剪枝(regular) | 6.0M |24.0MB  |73.6%  |6.55ms  |0.7778  | 0|
| normal剪枝(regular\max) | 6.0M |23.9MB  |73.6%  |6.42ms  |0.7779  | -3|




#### vgg16_bn_ssd300_fpga(without l2、dilated conv)
| 模型 | 参数量 |模型体积  |压缩率  |耗时  |mAP  |层数  |
| --- | --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |5.60ms  |0.7820  |  0|
| normal剪枝| 6.1M |24.5MB  |72.9%  |4.26ms  |0.7725  |   0|
| normal剪枝(max)| 6.1M |24.5MB  |72.9%  |4.10ms  |0.7726  |   -3|
| normal剪枝(regular) | 8.8M |35.2MB  |61.2%  |4.79ms  |0.7807  |   0|
| normal剪枝(regular\max) | 8.8M |35.2MB  |61.2%  |4.58ms  |0.7808  |   -3|


## Quantization(dorefa)+Pruning
首尾层不量化
### 先量化后剪枝
#### vgg16_bn_ssd300   
| 模型 | 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |0.7815  |   0|
| W8A8 | 22.7M |25.0MB  |72.4%  |0.7944  | 0|
| W8A8 normal剪枝(max\regular) | 5.7M |6.2MB  |93.2%  | 0.7862  | -3|


#### vgg16_bn_ssd300_fpga(without l2、dilated conv)     
| 模型| 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
|Baseline(300)| 22.7M |90.6MB  |0%  |0.7820  |   0|
| W8A8| 22.7M |25.0MB  |72.4%  |0.7865  | 0|
| W8A8 normal剪枝(max\regular)| 8.5M |9.0MB  |90.1%  | 0.7830  | -3|

### 先剪枝后量化
#### vgg16_bn_ssd300   
| 模型 | 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |0.7815  |   0|
| normal剪枝(max\regular) | 6.0M |23.9MB  |73.6%  |0.7779  | -3|
| normal剪枝(max\regular)W8A8 | 6.0M |7.0MB  |92.3%  | 0.7788 | -3|


#### vgg16_bn_ssd300_fpga(without l2、dilated conv) 
| 模型 | 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |0.7820  |   0|
| normal剪枝(max\regular) | 8.8M |35.2MB  |61.2%  |0.7808  |   -3|
| normal剪枝(max\regular)W8A8 | 8.8M |9.5MB  |89.5%  |0.7851  | -3|




## Quantization(IAO:BN训练中融合)+Pruning
首尾层不量化
### 先量化后剪枝
#### vgg16_bn_ssd300   
| 模型 | 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
| Baseline(300)| 22.7M |90.6MB  |0%  |0.7815  |   0|
| W8A8 | 22.7M |25.0MB  |72.4%  |0.7944  | 0|
| W8A8 normal剪枝(max\regular) | 5.7M |6.2MB  |93.2%  | 0.7862  | -3|


#### vgg16_bn_ssd300_fpga(without l2、dilated conv) 
| 模型| 参数量 |模型体积  |压缩率  |mAP  |层数  |
| --- | --- | --- | --- | --- | ---|
|Baseline(300)| 22.7M |90.6MB  |0%  |0.7820  |   0|
| W8A8| 22.7M |25.0MB  |72.4%  |0.7834  | 0|
| W8A8 normal剪枝(max\regular)| 8.5M |9.0MB  |90.1%  | 0.7830  | -3|














