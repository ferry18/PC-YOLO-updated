import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # D:\\way\\yolo\\yolo11\\ultralytics\\runs\\train\\exp8\\weights\\best.pt
    model = YOLO("./runs/train/exp14/weights/best.pt")
    model.predict(source="D:\\way\\yolo\\Datasets\\val",
                  project='runs/detect',
                  name='exp',
                  conf=0.1,
                  save=True  # 是否要保存结果图片
                  )
