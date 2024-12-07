import torch
from ultralytics import YOLO
import os

# 确保将自定义的 ACConv2d 层导入
from ultralytics.nn.modules import ACConv2d  # 你的自定义模块应该包含 ACConv2d 的定义

# 定义路径
model_path = 'D:/way/yolo/yolo11_2/ultralytics/runs/train/exp_datasets7_ACConv2d/weights/best.pt'
data_path = 'D:/way/yolo/Datasets/datasets/cityscapes.yaml'

# 验证函数
def validate_model(model_path, data_path):
    try:
        # 使用 YOLO 的验证功能
        model = YOLO(model_path)  # 加载模型
        model.val(data=data_path)  # 进行验证
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 检查模型和数据文件是否存在
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
    elif not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
    else:
        validate_model(model_path, data_path)
