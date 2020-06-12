# Author:LiPu
import numpy as np
import pandas as pd
import os

#聚类到都在这些框大小附近
class Kmeans:

    def __init__(self, cluster_number, data_file, anchors_file,input_shape):
        self.cluster_number = cluster_number  # 6 or 9
        self.data_file = data_file
        self.train_txt=data_file+'/train.txt'
        self.anchors_file = anchors_file
        self.input_shape=input_shape
    def iou(self, boxes, clusters):  # n box -> k clusters, boxes代表实际的每个box的宽和高， clusters为聚类中心
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]  # box的面积长x宽

        box_area = box_area.repeat(k)  # 将每个box的面积重复6次，构成一个n行6列的数组

        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]  # 计算每个聚类中心的面积
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))  # 将聚类中心构成与输入boxes具有同样维度的矩阵，可以避免循环运算

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]       #n,2
        # print(box_number)
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters从原来的众多box中随机选取6个box
        while True:

            distances = 1 - self.iou(boxes, clusters)#n,k

            current_nearest = np.argmin(distances, axis=1)#n--->每一个对应距离最小的cluster
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)   #找到聚为一堆的中位数

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):  # 将data保存进txt文档中，覆盖方式
        f = open(self.anchors_file, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])  # 若data只有1行，直接两个数据用逗号隔开保存
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])  # 若data有多行，就将每行数据以空格隔开
            f.write(x_y)
        f.close()

    def txt2boxes(self):

        with open(self.train_txt, 'r') as f:
            labels_files = [x.replace('data', str(self.data_file)).replace('/', os.sep).replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in f.read().splitlines()]
        dataSet = []
        for i in labels_files:
            f = open(i, 'r')
            for line in f:
                infos = line.split(" ")
                width = int(float(infos[3]) * self.input_shape[0])
                height = int(float(infos[4]) * self.input_shape[1])
                dataSet.append([width, height])
            f.close()


        result = np.array(dataSet)
        return result

    def csv2boxes(self):
        dataSet = []
        train_data = pd.read_csv(self.data_file, header=None)
        for value_i in train_data.values:
            end_num = len(value_i)
            for i in range(1, end_num, 5):
                width = value_i[i + 2] - value_i[i]
                height = value_i[i + 3] - value_i[i + 1]
                dataSet.append([width, height])  ##获取所有box的宽和高
        result = np.array(dataSet)
        return result

    def txt2clusters(self):
        # all_boxes = self.csv2boxes()  # 获取所有box的宽和高
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]  # 对于聚类结果，按照第一维度进行从小到大排序
        self.result2txt(result)   #写入txt
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number =12  # tiny-yolo--6, yolo--9     ssd-->n
    data_file = "/home/xpt/data/hand_dataset"
    anchors_file = "anchors.txt"
    input_shape=[300,300]#yolo-->(256,256) (416,416) (608,608)   ssd-->(300,300)  (500,500)  w,h
    kmeans = Kmeans(cluster_number=cluster_number, data_file=data_file, anchors_file=anchors_file,input_shape=input_shape)
    kmeans.txt2clusters()
