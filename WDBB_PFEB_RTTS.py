import warnings
import time
warnings.filterwarnings('ignore')

from ultralytics import YOLO
if __name__ == '__main__':
    # 记录训练开始时间
    start_time = time.time()
    weights = r'D:\way\yolo\yolo11\yolo11_2\ultralytics\runs\train\exp_hazy_WDBB_PFEB\weights\best.pt'
    # model = YOLO(r'ultralytics/cfg/models/11/WDBB_PFEB.yaml', weights=weights)
    model = YOLO(weights)
    # model.train(data=r'/mnt/result2/datasets/cityscapes.yaml',
    # model.train(data=r'D:\way\yolo\Datasets\Normal_to_Foggy\voc.yaml',
    # model.train(data=r'D:\\way\\yolo\\Datasets\\cityscapes-2\\city.yaml',
    # model.train(data=r'/mnt/datasets/cityscapes.yaml',
    # model.train(data=r'D:/way/yolo/Datasets/datasets_2/cityscapes.yaml',
    # model.train(data=r'/mnt/RTTS/dataset.yaml',
    # model.train(data=r'D:\way\yolo\Datasets\datasets\rain.yaml',
    model.train(data=r"D:\way\yolo\Datasets\BigRain2\cityscapes.yaml",
    cache=False,
    imgsz=640,
    epochs=20,
    single_cls=False,
    batch=8,
    close_mosaic=10,
    workers=0,   # win电脑最好设置为0，否则容易报线程错误
    device='cpu',
    optimizer='SGD',
    amp=True,
    project='runs/train',
    name='exp_BigRain_WDBB_PFEB',
    half=True)


