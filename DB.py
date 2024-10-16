import torch
from ultralytics import YOLO
import time
import os

# 定义模型路径
original_model_path = "D:/way/yolo/yolo11/ultralytics/runs/train/exp/weights/best.pt"  # 原始模型路径
improved_model_path = "D:/way/yolo/yolo11_2/ultralytics/runs/train/exp15/weights/best.pt"  # 改进模型路径
data_path = "D:/way/yolo/Datasets/datasets/cityscapes.yaml"  # 数据集配置文件

# 定义验证函数
def validate_model(model_path):
    model = YOLO(model_path)  # 加载模型
    start_time = time.time()
    results = model.val(data=data_path)  # 验证
    inference_time = time.time() - start_time  # 计算推理时间
    return results, inference_time

# 定义获取模型大小的函数
def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)  # 返回模型大小 (MB)

# 验证原始模型
original_results, original_inference_time = validate_model(original_model_path)
original_model_size = get_model_size(original_model_path)

# 验证改进模型
improved_results, improved_inference_time = validate_model(improved_model_path)
improved_model_size = get_model_size(improved_model_path)

# 输出结果
print("\n比较结果：")
print("原始模型验证结果：", original_results.results_dict)  # 输出原始模型的所有结果
print("改进模型验证结果：", improved_results.results_dict)  # 输出改进模型的所有结果

# 输出推理时间和模型大小
print(f"原始模型推理时间: {original_inference_time:.2f} 秒")
print(f"改进模型推理时间: {improved_inference_time:.2f} 秒")

print(f"原始模型大小: {original_model_size:.2f} MB")
print(f"改进模型大小: {improved_model_size:.2f} MB")

# 计算推理时间差
inference_time_diff = improved_inference_time - original_inference_time
if inference_time_diff < 0:
    print(f"改进模型比原始模型快 {abs(inference_time_diff):.2f} 秒。")
else:
    print(f"改进模型比原始模型慢 {inference_time_diff:.2f} 秒。")

# 计算模型大小差
model_size_diff = improved_model_size - original_model_size
if model_size_diff < 0:
    print(f"改进模型比原始模型小 {abs(model_size_diff):.2f} MB。")
else:
    print(f"改进模型比原始模型大 {model_size_diff:.2f} MB。")
