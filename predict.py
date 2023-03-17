import numpy as np
import torch
from PIL import ImageFont, ImageDraw
from cv2 import cv2
from matplotlib import pyplot as plt
# import cv2
from torchvision.transforms import ToTensor

from draw_rectangle import draw
from new_resnet import resnet50

# voc数据集的类别信息，这里转换成字典形式
classes = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

# 测试图片的路径
img_root = "D:\\program\\Object_detection\\YOLOV1-pytorch-main\\test.jpg"
# 网络模型
model = resnet50()
# 加载权重，就是在train.py中训练生成的权重文件yolo.pth
model.load_state_dict(torch.load("D:\\program\\Object_detection\\YOLOV1-pytorch-main\\yolo.pth"))
# 测试模式
model.eval()
# 设置置信度
confident = 0.2
# 设置iou阈值
iou_con = 0.4

# 类别信息，这里写的都是voc数据集的，如果是自己的数据集需要更改
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
# 类别总数：20
CLASS_NUM = len(VOC_CLASSES)

"""
注意：预测和训练的时候是不同的，训练的时候是有标签参考的，在测试的时候会直接输出两个预测框，
保留置信度比较大的，再通过NMS处理得到最终的预测结果，不要跟训练阶段搞混了
"""

# target 7*7*30  值域为0-1
class Pred():
    # 参数初始化
    def __init__(self, model, img_root):
        self.model = model
        self.img_root = img_root

    def result(self):
        # 读取测试的图像
        img = cv2.imread(self.img_root)
        # 获取高宽信息
        h, w, _ = img.shape
        # 调整图像大小
        image = cv2.resize(img, (448, 448))
        # CV2读取的图像是BGR，这里转回RGB模式
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图像均值
        mean = (123, 117, 104)  # RGB
        # 减去均值进行标准化操作
        img = img - np.array(mean, dtype=np.float32)
        # 创建数据增强函数
        transform = ToTensor()
        # 图像转为tensor，因为cv2读取的图像是numpy格式
        img = transform(img)
        # 输入要求是BCHW，加一个batch维度
        img = img.unsqueeze(0)
        # 图像输入模型，返回值为1*7*7*30的张量
        Result = self.model(img)
        # 获取目标的边框信息
        bbox = self.Decode(Result)
        # 非极大值抑制处理
        bboxes = self.NMS(bbox)    # n*6   bbox坐标是基于7*7网格需要将其转换成448
        # draw(image, bboxes, classes)
        if len(bboxes) == 0:
            print("未识别到任何物体")
            print("尝试减小 confident 以及 iou_con")
            print("也可能是由于训练不充分，可在训练时将epoch增大")        
        for i in range(0, len(bboxes)):    # bbox坐标将其转换为原图像的分辨率
            bboxes[i][0] = bboxes[i][0] * 64
            bboxes[i][1] = bboxes[i][1] * 64
            bboxes[i][2] = bboxes[i][2] * 64
            bboxes[i][3] = bboxes[i][3] * 64

            x1 = bboxes[i][0].item()    # 后面加item()是因为画框时输入的数据不可一味tensor类型
            x2 = bboxes[i][1].item()
            y1 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            class_name = bboxes[i][5].item()
            print(x1, x2, y1, y2, VOC_CLASSES[int(class_name)])

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (144, 144, 255))   # 画框
            plt.imsave('test_001.jpg', image)
        # cv2.imwrite("img", image)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

    # 接受的result的形状为1*7*7*30
    def Decode(self, result):
        # 去掉batch维度
        result = result.squeeze()
        # 提取置信度信息，并在最后一个维度增加一维，跟后面匹配result[:, :, 4]的形状为7*7,最后变为7*7*1
        grid_ceil1 = result[:, :, 4].unsqueeze(2)
        # 同上
        grid_ceil2 = result[:, :, 9].unsqueeze(2)
        # 两个置信度信息按照维度2拼接
        grid_ceil_con = torch.cat((grid_ceil1, grid_ceil2), 2)
        # 按照第二个维度进行最大值求取，一个grid ceil两个bbox，两个confidence，也就是找置信度比较大的那个，形状都是7*7
        grid_ceil_con, grid_ceil_index = grid_ceil_con.max(2)
        # 找出一个gird cell中预测类别最大的物体的索引和预测条件类别概率
        class_p, class_index = result[:, :, 10:].max(2)
        # 计算出物体的真实概率，类别最大的物体乘上置信度比较大的那个框得到最终的真实物体类别概率
        class_confidence = class_p * grid_ceil_con
        # 定义一个张量，记录位置信息
        bbox_info = torch.zeros(7, 7, 6)
        for i in range(0, 7):
            for j in range(0, 7):
                # 获取置信度比较大的索引位置
                bbox_index = grid_ceil_index[i, j]
                # 把置信度比较大的那个框的位置信息保存到bbox_info中，另一个直接抛弃
                bbox_info[i, j, :5] = result[i, j, (bbox_index * 5):(bbox_index+1) * 5]
        # 真实目标概率
        bbox_info[:, :, 4] = class_confidence
        # 类别信息
        bbox_info[:, :, 5] = class_index
        # 返回预测的结果,7*7*6    6 = bbox4个信息+类别概率+类别代号
        return bbox_info

    # 非极大值抑制处理，按照类别处理，bbox为Decode获取的预测框的位置信息和类别概率和类别信息
    def NMS(self, bbox, iou_con=iou_con):
        for i in range(0, 7):
            for j in range(0, 7):
                # xc = bbox[i, j, 0]
                # yc = bbox[i, j, 1]
                # w = bbox[i, j, 2] * 7
                # h = bbox[i, j, 3] * 7
                # Xc = i + xc
                # Yc = j + yc
                # xmin = Xc - w/2
                # xmax = Xc + w/2
                # ymin = Yc - h/2
                # ymax = Yc + h/2
                # 注意，目前bbox的四个坐标是以grid ceil的左上角为坐标原点，而且单位不一致中心点偏移的坐标是归一化为(0-7),宽高是(0-7),，单位不一致，全部归一化为(0-7)
                # 计算预测框的左上角右下角相对于7*7网格的位置
                xmin = j + bbox[i, j, 0] - bbox[i, j, 2] * 7 / 2     # xmin
                xmax = j + bbox[i, j, 0] + bbox[i, j, 2] * 7 / 2     # xmax
                ymin = i + bbox[i, j, 1] - bbox[i, j, 3] * 7 / 2     # ymin
                ymax = i + bbox[i, j, 1] + bbox[i, j, 3] * 7 / 2     # ymax

                bbox[i, j, 0] = xmin
                bbox[i, j, 1] = xmax
                bbox[i, j, 2] = ymin
                bbox[i, j, 3] = ymax
        # 调整形状，bbox本来就是(49*6)，这里感觉没必要
        bbox = bbox.view(-1, 6)
        # 存放最终需要保留的预测框
        bboxes = []
        # 取出每个gird cell中的类别信息，返回一个列表
        ori_class_index = bbox[:, 5]
        # 按照类别进行排序，从高到低，返回的是排序后的类别列表和对应的索引位置,如下：
        """
        类别排序
        tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
         3.,  3.,  3.,  4.,  4.,  4.,  4.,  5.,  5.,  5.,  6.,  6.,  6.,  6.,
         6.,  6.,  6.,  6.,  7.,  8.,  8.,  8.,  8.,  8., 14., 14., 14., 14.,
        14., 14., 14., 15., 15., 16., 17.], grad_fn=<SortBackward0>)
        位置索引
        tensor([48, 47, 46, 45, 44, 43, 42,  7,  8, 22, 11, 16, 14, 15, 24, 20,  1,  2,
         6,  0, 13, 23, 25, 27, 32, 39, 38, 35, 33, 31, 30, 28,  3, 26, 10, 19,
         9, 12, 29, 41, 40, 21, 37, 36, 34, 18, 17,  5,  4])
        """
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        # class_index是一个tensor，这里把他转为列表形式
        class_index = class_index.tolist()
        # 根据排序后的索引更改bbox排列顺序
        bbox = bbox[class_order, :]

        a = 0
        for i in range(0, CLASS_NUM):
            # 统计目标数量，即某个类别出现在grid cell中的次数
            num = class_index.count(i)
            # 预测框中没有这个类别就直接跳过
            if num == 0:
                continue
            # 提取同一类别的所有信息
            x = bbox[a:a+num, :]
            # 提取真实类别概率信息
            score = x[:, 4]
            # 提取出来的某一类别按照真实类别概率信息高度排序，递减
            score_index, score_order = score.sort(dim=0, descending=True)
            # 根据排序后的结果更改真实类别的概率排布
            y = x[score_order, :]
            # 先看排在第一位的物体的概率是否大有给定的阈值，不满足就不看这个类别了，丢弃全部的预测框
            if y[0, 4] >= confident:
                for k in range(0, num):
                    # 真实类别概率，排序后的
                    y_score = y[:, 4]
                    # 对真实类别概率重新排序，保证排列顺序依照递减，其实跟上面一样的，多此一举
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    # 判断概率是否大于0
                    if y[k, 4] > 0:
                        # 计算预测框的面积
                        area0 = (y[k, 1] - y[k, 0]) * (y[k, 3] - y[k, 2])
                        for j in range(k+1, num):
                            # 计算剩余的预测框的面积
                            area1 = (y[j, 1] - y[j, 0]) * (y[j, 3] - y[j, 2])
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 1], y[j, 1])
                            y1 = max(y[k, 2], y[j, 2])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = 0
                                h = 0
                            inter = w * h
                            # 计算与真实目标概率最大的那个框的iou
                            iou = inter / (area0 + area1 - inter)
                            # iou大于一定值则认为两个bbox识别了同一物体删除置信度较小的bbox
                            # 同时物体类别概率小于一定值也认为不包含物体
                            if iou >= iou_con or y[j, 4] < confident:
                                y[j, 4] = 0
                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])
            # 进入下个类别
            a = num + a
        #     返回最终预测的框
        return bboxes


if __name__ == "__main__":
    Pred = Pred(model, img_root)
    Pred.result()

