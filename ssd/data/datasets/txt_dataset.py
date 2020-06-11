# !/usr/bin/env python
# coding:utf-8
# Author:XuPengTao
# Date: 2020/3/10
import torch.utils.data
import numpy as np
from PIL import Image
import os
from ssd.structures.container import Container
from ssd.utils.parse_config import parse_data_cfg,load_classes
class TxtDataset(torch.utils.data.Dataset):
    def __init__(self,
                 transform=None, target_transform=None,
                 dataset_type="train",dataset_name='oxfordhand'):
        data_path=os.path.join('configs',dataset_name+'.data')
        self.data=parse_data_cfg(data_path)
        classes = load_classes(self.data['names'])
        self.class_names = ['BACKGROUND'] + classes
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type
        path =self.data[dataset_type] #txt
        with open(path, 'r') as f:
            self.img_files = [os.path.join(self.data['images'],x) for x in f.read().splitlines()]
        with open(path, 'r') as f:
            self.label_files = [(os.path.join(self.data['labels'],x)).replace(os.path.splitext(x)[-1], '.txt')
                       for x in f.read().splitlines()]
        self.num = len(self.img_files)
        self.ids = [str(i) for i in range(self.num)]

    def __getitem__(self, index):
        boxes, labels, is_difficult = self._get_annotation(index)
        image = self._read_image(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(index)
    def _get_annotation(self, index):
        image = self._read_image(index)
        h, w, channels = image.shape
        # Load labels
        boxes = []  # ["XMin", "YMin", "XMax", "YMax"]
        labels = []
        with open(self.label_files[index], 'r') as f:
            x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            #   [[ label_ind,xmid, ymid, w, h], ... ]
        if x.size > 0:
            # Normalized xywh to norm xyxy format
            boxes = x.copy()[:, :4]
            boxes[:, 0] = x[:, 1] - x[:, 3] / 2
            boxes[:, 1] = x[:, 2] - x[:, 4] / 2
            boxes[:, 2] = x[:, 1] + x[:, 3] / 2
            boxes[:, 3] = x[:, 2] + x[:, 4] / 2
            labels = x[:, 0] + 1  # +1跳过背景类  对于ssd
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        labels = np.array(labels, dtype='int64')# make labels 64 bits to satisfy the cross_entropy function
        return boxes, labels, is_difficult

    def get_image(self, index):
        image = self._read_image(index)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def __len__(self):
        return self.num

    def get_img_info(self,index):
        image = self._read_image(index)
        h, w, channels = image.shape
        return {"height": h, "width": w}
    def _read_image(self, index):
        image_file =self.img_files[index]
        image =Image.open(image_file).convert("RGB")
        image=np.array(image)
        return image

