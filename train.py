import warnings
import time
warnings.filterwarnings('ignore')

from ultralytics import YOLO
if __name__ == '__main__':
    # 记录训练开始时间
    start_time = time.time()
    model = YOLO(r'ultralytics/cfg/models/11/PC_YOLO.yaml')
    # 数据集.yaml路径
    model.train(data=r"./Datasets/BigRain2/cityscapes.yaml",
    cache=False,
    imgsz=640,
    epochs=20,
    single_cls=False,
    batch=8,
    close_mosaic=10,
    workers=0,
    device='cpu',
    optimizer='SGD',
    amp=True,
    project='runs/train',
    name='exp_',
    half=True)


