import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/train/exp_RTTS_PC_YOLO/weights/best.pt")
    model.predict(source="D:\way\yolo\img\detect\BigRain",
                  project='runs/detect',
                  name='exp_',
                  conf=0.5,
                  save=True  # 是否要保存结果图片
                  )
