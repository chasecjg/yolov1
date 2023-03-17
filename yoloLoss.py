import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

warnings.filterwarnings('ignore')  # 忽略警告消息
CLASS_NUM = 20    # （使用自己的数据集时需要更改）

class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        # 一般而言 l_coord = 5 ， l_noobj = 0.5
        super(yoloLoss, self).__init__()
        # 网格数
        self.S = S  # S = 7
        # bounding box数量
        self.B = B  # B = 2
        # 权重系数
        self.l_coord = l_coord
        # 权重系数
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):  # box1(2,4)  box2(1,4)
        N = box1.size(0)  # 2
        M = box2.size(0)  # 1

        lt = torch.max(  # 返回张量所有元素的最大值
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        rb = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  重复面积

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        # iou的形状是(2,1)，里面存放的是两个框的iou的值
        return iou  # [2,1]

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,7,7,30)
        target_tensor: (tensor) size(batchsize,7,7,30)，就是在yoloData中制作的标签
        '''
        # batchsize大小
        N = pred_tensor.size()[0]
        # 判断目标在哪个网格，输出B*7*7的矩阵，有目标的地方为True，其他地方为false,这里就用了第4位，第9位和第4位一样就没判断
        coo_mask = target_tensor[:, :, :, 4] > 0
        # 判断目标不在那个网格，输出B*7*7的矩阵，没有目标的地方为True，其他地方为false
        noo_mask = target_tensor[:, :, :, 4] == 0
        # 将 coo_mask tensor 在最后一个维度上增加一维，并将其扩展为与 target_tensor tensor 相同的形状，得到含物体的坐标等信息，大小为batchsize*7*7*30
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        # 将 noo_mask 在最后一个维度上增加一维，并将其扩展为与 target_tensor tensor 相同的形状，得到不含物体的坐标等信息，大小为batchsize*7*7*30
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # 根据label的信息从预测的张量取出对应位置的网格的30个信息按照出现序号拼接成以一维张量，这里只取包含目标的
        coo_pred = pred_tensor[coo_mask].view(-1, int(CLASS_NUM + 10))
        # 所有的box的位置坐标和置信度放到box_pred中，塑造成X行5列（-1表示自动计算），一个box包含5个值
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        # 类别信息
        class_pred = coo_pred[:, 10:]  # [n_coord, 20]

        # pred_tensor[coo_mask]把pred_tensor中有物体的那个网格对应的30个向量拿出来，这里对应的是label向量，只计算有目标的
        coo_target = target_tensor[coo_mask].view(-1, int(CLASS_NUM + 10))
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 不包含物体grid ceil的置信度损失,这里是label的输出的向量。
        noo_pred = pred_tensor[noo_mask].view(-1, int(CLASS_NUM + 10))
        noo_target = target_tensor[noo_mask].view(-1, int(CLASS_NUM + 10))
        # 创建一个跟noo_pred相同形状的张量，形状为（x,30），里面都是全0或全1，再使用bool将里面的0或1转为true和false
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()
        # 把创建的noo_pred_mask全部改成false，因为这里对应的是没有目标的张量
        noo_pred_mask.zero_()
        # 把不包含目标的张量的置信度位置置为1
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1

        # 跟上面的pred_tensor[coo_mask]一个意思，把不包含目标的置信度提取出来拼接成一维张量
        noo_pred_c = noo_pred[noo_pred_mask]
        # 同noo_pred_c
        noo_target_c = noo_target[noo_pred_mask]
        # 计算loss，让预测的值越小越好，因为不包含目标，置信度越为0越好
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)  # 均方误差
        # 注意：上面计算的不包含目标的损失只计算了置信度，其他的都没管
        """
        计算包含目标的损失：位置损失+类别损失
        """
        # 先创建两个张量用于后面匹配预测的两个编辑框：一个负责预测，一个不负责预测
        # 创建一跟box_target相同的张量,这里用来匹配后面负责预测的框
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        # 全部置为False
        coo_response_mask.zero_()  # 全部元素置False

        # 创建一跟box_target相同的张量，这里用来匹配不负责预测的框
        no_coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        # 全部置为False
        no_coo_response_mask.zero_()

        # 创建一个全0张量，匹配后面的预测框的iou
        box_target_iou = torch.zeros(box_target.size()).cuda()

        # 遍历每一个标注框，每次遍历两个是因为一个标注框对应有两个预测框要跟他匹配，box1 = 预测框  box2 = ground truth
        # box_target.size()[0]：有多少bbox，并且一次取两个bbox，因为两个bbox才是一个完整的预测框
        for i in range(0, box_target.size()[0], 2):  #
            #  第i个grid ceil对应的两个bbox
            box1 = box_pred[i:i + 2]
            # 创建一个和box1大小(2,5)相同的浮点型张量用来存储坐标，这里代码使用的torch版本可能比较老，其实Variable可以省略的
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # box1_xyxy[:, :2]为预测框中心点坐标相对于所在grid cell左上角的偏移，前面在数据处理的时候讲过label里面的值是归一化为(0-7)的，
            # 因此这里得反归一化成跟宽高一样比例，归一化到(0-1),减去宽高的一半得到预测框的左上角的坐标相对于他所在的grid cell的左上角的偏移量
            box1_xyxy[:, :2] = box1[:, :2] / float(self.S) - 0.5 * box1[:, 2:4]
            # 计算右下角坐标相对于预测框所在的grid cell的左上角的偏移量
            box1_xyxy[:, 2:4] = box1[:, :2] / float(self.S) + 0.5 * box1[:, 2:4]
            # target中的两个框的目标信息是一模一样的，这里取其中一个就行了
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / float(self.S) - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / float(self.S) + 0.5 * box2[:, 2:4]
            # 计算两个预测框与标注框的IoU值，返回计算结果，是个列表
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            # 通过max()函数获取与gt最大的框的iou和索引号(第几个框)
            max_iou, max_index = iou.max(0)
            # 将max_index放到GPU上
            max_index = max_index.data.cuda()
            # 保留IoU比较大的那个框
            coo_response_mask[i + max_index] = 1  # IOU最大的bbox
            # 舍去的bbox,两个框单独标记为1，分开存放，方便下面计算
            no_coo_response_mask[i + 1 - max_index] = 1
            # 将预测框比较大的IoU的那个框保存在box_target_iou中，
            # 其中i + max_index表示当前预测框对应的位置，torch.LongTensor([4]).cuda()表示在box_target_iou中存储最大IoU的值的位置。
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        # 放到GPU上
        box_target_iou = Variable(box_target_iou).cuda()
        # 负责预测物体的预测框的位置信息（含物体的grid ceil的两个bbox与ground truth的IOU较大的一方）
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        # 标注信息，拿出来一个计算就行了，因为两个信息一模一样
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        # IOU较小的一方
        no_box_pred_response = box_pred[no_coo_response_mask].view(-1, 5)
        no_box_target_response_iou = box_target_iou[no_coo_response_mask].view(-1, 5)
        # 不负责预测物体的置信度置为0，本来就是0，这里有点多此一举
        no_box_target_response_iou[:, 4] = 0
        # 负责预测物体的标注框对应的label的信息
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        # 包含物的体grid ceil中IOU较大的bbox置信度损失
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        # 不包含物体的grid ceil中舍去的bbox的置信度损失
        no_contain_loss = F.mse_loss(no_box_pred_response[:, 4], no_box_target_response_iou[:, 4], size_average=False)
        # 负责预测物体的预测框的位置损失
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)

        # 负责预测物体的所在grid cell的类别损失
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        # 计算总损失，这里有个权重
        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * (nooobj_loss + no_contain_loss) + class_loss) / N
