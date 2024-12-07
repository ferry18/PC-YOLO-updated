import time
import torch
from ultralytics import YOLO
import cv2  # OpenCV for video/image input

# 直接在代码中指定数据集路径和模型路径
dataset_path = 'D:\way\yolo\Datasets\datasets\\rain.yaml'
model_path = 'D:\way\yolo\yolo11\yolo11_2\\ultralytics\\runs\\train\exp_RTTS_WDBB_PFEB\weights\\best.pt'

print(f"Using dataset: {dataset_path}")
print(f"Using model: {model_path}")

# 加载YOLO模型
model = YOLO(model_path)  # 加载训练好的YOLO模型

# 测试推理函数，计算FPS
def calculate_fps(image_path):
    # 读取图片
    img = cv2.imread(image_path)  # 使用OpenCV读取图片
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # 预处理图片并开始计时
    start_time = time.time()  # 开始计时
    results = model(img)  # 进行推理
    end_time = time.time()  # 结束计时

    # 计算FPS
    inference_time = end_time - start_time
    fps = 1.0 / inference_time  # FPS = 1 / 推理时间
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")

    # 可选：绘制推理结果（例如在图片上绘制检测框）
    results.show()  # 显示推理结果，通常会在新窗口显示图片

# 测试图片路径（您可以替换为您自己的图片路径）
image_path = 'D:\way\yolo\Datasets\datasets\images\\test\\bielefeld\\bielefeld_000000_000321_leftImg8bit.png'

# 计算FPS
calculate_fps(image_path)
