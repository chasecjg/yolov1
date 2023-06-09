import pandas as pd
import torch
import cv2
import os
import os.path
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image

pd.set_option('display.max_rows', None)  # ÏÔÊ¾È«²¿ÐÐ
pd.set_option('display.max_columns', None)  # ÏÔÊ¾È«²¿ÁÐ

# 根据txt文件制作ground truth
CLASS_NUM = 20  # 使用其他训练集需要更改


class yoloDataset(Dataset):
    image_size = 448  # 输入图片大小

    # list_file为txt文件  img_root为图片路径
    def __init__(self, img_root, list_file, train, transform):
        # 初始化参数
        self.root = img_root
        self.train = train
        self.transform = transform
        # 后续要提取txt文件的信息，分类后装入以下三个列表
        # 文件名
        self.fnames = []
        # 位置信息
        self.boxes = []
        # 类别信息
        self.labels = []

        # 网格大小
        self.S = 7
        # 候选框个数
        self.B = 2
        # 类别数目
        self.C = CLASS_NUM
        # 求均值用的
        self.mean = (123, 117, 104)
        # 打开文件，就是voctrain.txt或者voctest.txt文件
        file_txt = open(list_file)
        # 读取txt文件每一行
        lines = file_txt.readlines()
        # 逐行开始操作
        for line in lines:
            # 去除字符串开头和结尾的空白字符，然后按照空白字符（包括空格、制表符、换行符等）分割字符串并返回一个列表
            splited = line.strip().split()
            # 存储图片的名字
            self.fnames.append(splited[0])
            # 计算一幅图片里面有多少个bbox，注意voctrain.txt或者voctest.txt一行数据只有一张图的信息
            num_boxes = (len(splited) - 1) // 5
            # 保存位置信息
            box = []
            # 保存标签信息
            label = []
            # 提取坐标信息和类别信息
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                # 提取类别信息，即是20种物体里面的哪一种  值域 0-19
                c = splited[5 + 5 * i]
                # 存储位置信息
                box.append([x, y, x2, y2])
                # 存储标签信息
                label.append(int(c))
            # 解析完所有行的信息后把所有的位置信息放到boxes列表中，boxes里面的是每一张图的坐标信息，也是一个个列表，即形式是[[[x1,y1,x2,y2],[x3,y3,x4,y4]],[[x5,y5,x5,y6]]...]这样的
            self.boxes.append(torch.Tensor(box))
            # 形式是[[1,2],[3,4]...]，注意这里是标签，对应位整型数据
            self.labels.append(torch.LongTensor(label))
        # 统计图片数量
        self.num_samples = len(self.boxes)


    def __getitem__(self, idx):
        # 获取一张图像
        fname = self.fnames[idx]
        # 读取这张图像
        img = cv2.imread(os.path.join(self.root + fname))
        # 拷贝一份，避免在对数据进行处理时对原始数据进行修改
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        """
        数据增强里面的各种变换用pytorch自带的transform是做不到的，因为对图片进行旋转、随即裁剪等会造成bbox的坐标也会发生变化，
        所以需要自己来定义数据增强,这里推荐使用功albumentations更简单便捷
        """
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            # img = self.RandomHue(img)
            # img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            # img, boxes, labels = self.randomCrop(img, boxes, labels)
        #     获取图像高宽信息
        h, w, _ = img.shape
        # 归一化位置信息，.expand_as(boxes)的作用是将torch.Tensor([w, h, w, h])扩展成和boxes一样的维度，这样才能进行后面的归一化操作
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 坐标归一化处理[0,1]，为了方便训练
        # cv2读取的图像是BGR，转成RGB
        img = self.BGR2RGB(img)
        # 减去均值，帮助网络更快地收敛并提高其性能
        img = self.subMean(img, self.mean)
        # 调整图像到统一大小
        img = cv2.resize(img, (self.image_size, self.image_size))
        # 将图片标签编码到7x7*30的向量，也就是我们yolov1的最终的输出形式，这个地方不了解的可以去看看yolov1原理
        target = self.encoder(boxes, labels)
        # 进行数据增强操作
        for t in self.transform:
            img = t(img)
        # 返回一张图像和所有标注信息
        return img, target

    def __len__(self):
        return self.num_samples

    # 编码图像标签为7x7*30的向量，输入的boxes为一张图的归一化形式位置坐标(X1,Y1,X2,Y2)
    def encoder(self, boxes, labels):
        # 网格大小
        grid_num = 7
        # 定义一个空的7*7*30的张量
        target = torch.zeros((grid_num, grid_num, int(CLASS_NUM + 10)))
        # 对网格进行归一化操作
        cell_size = 1. / grid_num  # 1/7
        # 计算每个边框的宽高，wh是一个列表，里面存放的是每一张图的标注框的高宽，形式为[[h1,w1,[h2,w2]...]
        wh = boxes[:, 2:] - boxes[:, :2]

        # 每一张图的标注框的中心点坐标，cxcy也是一个列表，形式为[[cx1,cy1],[cx2,cy2]...]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        # 遍历每个每张图上的标注框信息，cxcy.size()[0]为标注框个数，即计算一张图上几个标注框
        for i in range(cxcy.size()[0]):
            # 取中心点坐标
            cxcy_sample = cxcy[i]
            # 中心点坐标获取后，计算这个标注框属于哪个grid cell，因为上面归一化了，这里要反归一化，还原回去，坐标从0开始，所以减1，注意坐标从左上角开始
            ij = (cxcy_sample / cell_size).ceil() - 1
            # 把标注框框所在的gird cell的的两个bounding box置信度全部置为1，多少行多少列，多少通道的值置为1
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            # 把标注框框所在的gird cell的的两个bounding box类别置为1，这样就完成了该标注框的label信息制作了
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1

            # 预测框的中心点在图像中的绝对坐标（xy），归一化的
            xy = ij * cell_size
            # 标注框的中心点坐标与grid cell左上角坐标的差值，这里又变为了相对于（7*7）的坐标了，目的应该是防止梯度消失，因为这两个值减完后太小了
            delta_xy = (cxcy_sample - xy) / cell_size
            # 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例
            # 将目标框的宽高信息存储到target张量中对应的位置上
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]  # w1,h1
            # 目标框的中心坐标相对于所在的grid cell左上角的偏移量保存在target张量中对应的位置上，注意这里保存的是偏移量，并且是相对于（7*7）的坐标，而不是归一的1*1
            target[int(ij[1]), int(ij[0]), :2] = delta_xy  # x1,y1

            # 两个bounding box，这里同上
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # w2,h2
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # [5,7) 表示x2,y2
        # 返回制作的标签，可以看到，除了有标注框所在得到位置，其他地方全为0
        return target  # (xc,yc) = 7*7   (w,h) = 1*1

    # 以下方法都是数据增强操作，不介绍了，需要注意的是有些增强操作会让图片变形，如仿射变换，这个时候对应的标签也要跟着变化

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):,
                int(shift_x):,
                :] = bgr[:height - int(shift_y),
                     :width - int(shift_x),
                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y),
                int(shift_x):,
                :] = bgr[-int(shift_y):,
                     :width - int(shift_x),
                     :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width +
                                                  int(shift_x), :] = bgr[:height -
                                                                          int(shift_y), -
                                                                                        int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(
                    shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor(
                [[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor(
                [[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor(
                [[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

# def main():
#     file_root = 'VOCdevkit/VOC2007/JPEGImages/'
#     train_dataset = yoloDataset(
#         img_root=file_root,
#         list_file='voctrain.txt',
#         train=True,
#         transform=[
#             ToTensor()])
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=2,
#         drop_last=True,
#         shuffle=False,
#         num_workers=0)
#     train_iter = iter(train_loader)
#     for i in range(100):
#         img, target = next(train_iter)
#         print(img.shape)
#
#
# if __name__ == '__main__':
#     main()
