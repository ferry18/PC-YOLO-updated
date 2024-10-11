import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'./ultralytics/cfg/models/11/yolo11.yaml')
    # model.train(data=r'/mnt/result2/datasets/cityscapes.yaml',
    # model.train(data=r'D:\way\yolo\Datasets\Normal_to_Foggy\voc.yaml',
    model.train(data=r'/mnt/yolo11/datasets/cityscapes.yaml',
    cache=False,
    imgsz=640,
    epochs=200,
    single_cls=False,
    batch=8,
    close_mosaic=10,
    workers=0,   # win电脑最好设置为0，否则容易报线程错误
    device='0',
    optimizer='SGD',
    amp=True,
    project='runs/train',
    name='exp')
