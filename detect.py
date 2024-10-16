import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # D:\\way\\yolo\\yolo11\\ultralytics\\runs\\train\\exp8\\weights\\best.pt
    model = YOLO("./runs/train/exp15/weights/best.pt")
    # model = YOLO("yolo11n.pt")
    model.predict(source="D:\\way\\yolo\\Datasets\\val",
                  project='runs/detect',
                  name='exp',
                  conf=0.6,
                  save=True  # 是否要保存结果图片
                  )

# yolo val model=D:/way/yolo/yolo11_2/ultralytics/runs/train/exp15/weights/best.pt data=D:/way/yolo/Datasets/datasets/cityscapes.yaml

