import warnings
import time
import os
import torch  # 用于保存模型权重

warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    # 记录训练开始时间
    start_time = time.time()

    # 初始化模型
    model = YOLO(r'./ultralytics/cfg/models/11/FriendNet_YOLOv11.yaml')

    # 开始训练
    # model.train(data=r'/mnt/RTTS/dataset.yaml',
    # model.train(data=r'D:/way/yolo/Datasets/datasets_2/cityscapes.yaml',
    model.train(data=r'D:/way/yolo/Datasets/RTTS/dataset.yaml',
    # model.train(data=r'/mnt/RTTS/dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=8,
                close_mosaic=10,
                workers=0,  # win电脑最好设置为0，否则容易报线程错误
                device='cpu',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp_RTTS2_')

