import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df_yolov11 = pd.read_csv('D:\way\yolo\yolo11\yolo11_2\\ultralytics\\runs\\train\exp_RTTS_1\\results.csv')
df_yolo_wp = pd.read_csv('D:\way\yolo\yolo11\yolo11_2\\ultralytics\\runs\\train\exp_RTTS_WDBB_PFEB\\results.csv')

# 去除列名中的空格
df_yolov11.columns = df_yolov11.columns.str.strip()
df_yolo_wp.columns = df_yolo_wp.columns.str.strip()

# 绘制 mAP 50 图表
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df_yolov11['epoch'], df_yolov11['metrics/mAP50(B)'], label='YOLOv11', color='blue')
plt.plot(df_yolo_wp['epoch'], df_yolo_wp['metrics/mAP50(B)'], label='Ours', color='orange')
plt.xlabel('Epoch')
plt.ylabel('mAP 50')
plt.title('(a) mAP 50')
plt.legend()

# 绘制 mAP 50-95 图表
plt.subplot(1, 2, 2)
plt.plot(df_yolov11['epoch'], df_yolov11['metrics/mAP50-95(B)'], label='YOLOv11', color='blue')
plt.plot(df_yolo_wp['epoch'], df_yolo_wp['metrics/mAP50-95(B)'], label='Ours', color='orange')
plt.xlabel('Epoch')
plt.ylabel('mAP 50:95')
plt.title('(b) mAP 50:95')
plt.legend()

# 保存图表
plt.tight_layout()
plt.savefig('mAP_RTTS.png')  # 保存为 PNG 文件
plt.show()  # 显示图表