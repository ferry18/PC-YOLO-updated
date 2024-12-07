import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # D:\\way\\yolo\\yolo11\\ultralytics\\runs\\train\\exp8\\weights\\best.pt
    model = YOLO("runs/train/exp_rain_yolov11/weights/best.pt")
    # model = YOLO("yolo11n.pt")
    model.predict(source="D:\way\yolo\img\detect\BigRain",
                  project='runs/detect',
                  name='exp_v11_BigRain2',
                  conf=0.5,
                  save=True  # 是否要保存结果图片
                  )

# yolo val model=D:/way/yolo/yolo11_2/ultralytics/runs/train/exp_datasets7_ACConv2d/weights/best.pt data=D:/way/yolo/Datasets/datasets/cityscapes.yaml

