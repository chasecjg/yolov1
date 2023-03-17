# YOLOV1-pytorch
使用的pytorch版本为1.10

主干网络替换为了改进的ResNet50

代码对应博客地址：https://blog.csdn.net/qq_38683460/article/details/129578355?spm=1001.2014.3001.5501

VOC2007数据集下载地址：

链接：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

权重下载地址：

google drive: 

链接：https://drive.google.com/file/d/1hdtk2rnp2BFIQQDhXZpM8IDBS3eUqeby/view?usp=sharing

百度网盘：

链接: https://pan.baidu.com/s/1EHoic_cfUyrNmsoyUaM6EA 

提取码: yolo 



**一.训练**

**使用VOC2007数据集训练步骤：**

1、下载数据集放入文件夹中

2、运行write_txt.py生成训练以及测试需要用到的文本文件

3、运行train.py开始训练

**使用自己的训练集训练：**

1、将xml文件放入\VOCdevkit\VOC2007\Annotations

2、将jpg文件放入\VOCdevkit\VOC2007\JPEGImages

3、更改write_txt.py中的VOC_CLASSES

4、更改yoloData.py，yoloLoss.py与new_resnet.py中的CLASS_NUM

5、运行write_txt.py生成训练以及测试需要用到的文本文件

6、运行train.py开始训练


**二.测试**

1、测试文件为predict.py

2、加载直接训练好的权重文件放到model.load_state_dict(torch.load(your model weight path))，你的测试图片路径img_root

3、直接运行即可
