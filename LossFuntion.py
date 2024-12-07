import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df_yolo_wp = pd.read_csv('D:\way\yolo\yolo11\yolo11_2\\ultralytics\\runs\\train\exp_RTTS_WDBB_PFEB\\results.csv')
# df_yolo_wp = pd.read_csv('D:\way\yolo\yolo11\yolo11_2\\ultralytics\\runs\\train\exp_rain_WDBB_PFEB3\\results.csv')

# 去除列名中的空格
df_yolo_wp.columns = df_yolo_wp.columns.str.strip()

# 假设每个 epoch 有相同的迭代次数，这里以 100 为例
iterations_per_epoch = 100
total_iterations = df_yolo_wp['epoch'] * iterations_per_epoch

# 绘制 Loss_cls 和 Loss_bbox 随迭代次数变化的折线图
plt.figure(figsize=(8, 5))
plt.plot(total_iterations, df_yolo_wp['train/cls_loss'], label='Loss_cls', color='blue')
plt.plot(total_iterations, df_yolo_wp['train/box_loss'], label='Loss_bbox', color='orange')

plt.xlabel('Number of iters')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('loss_RTTS_yolo_wp.png')  # 保存为 PNG 文件
plt.show()  # 显示图表