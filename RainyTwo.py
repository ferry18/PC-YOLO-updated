import cv2
import numpy as np

def add_raindrop_effect(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # 图片的宽度和高度
    width, height = image.shape[1], image.shape[0]

    # 创建雨滴效果的单通道图层
    raindrop_mask = np.zeros((height, width), dtype=np.float32)
    num_drops = 20  # 雨滴数量
    max_drop_size = 600  # 最大雨滴大小
    min_drop_size = 50  # 最小雨滴大小

    for _ in range(num_drops):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        drop_size = np.random.randint(min_drop_size, max_drop_size)  # 随机雨滴大小
        decay_rate = np.random.uniform(0.5, 1.0)  # 随机透明度衰减率

        # 绘制半透明雨滴
        for i in range(-drop_size // 2, drop_size // 2 + 1):
            for j in range(-drop_size // 2, drop_size // 2 + 1):
                # 检查边界
                if 0 <= y + i < height and 0 <= x + j < width:
                    distance = (i ** 2 + j ** 2) ** 0.5
                    if distance <= drop_size // 2:
                        # 计算透明度，使用随机衰减率
                        opacity = 1 - (distance / (drop_size // 2)) ** decay_rate
                        raindrop_mask[y + i, x + j] = opacity

    # 调整雨滴图层的透明度
    raindrop_mask *= 1  # 降低透明度

    # 将雨滴图层与原图混合，模拟透明度
    raindrop_layer = (raindrop_mask * 255).astype(np.uint8)
    mixed_image = cv2.addWeighted(image, 1, cv2.cvtColor(raindrop_layer, cv2.COLOR_GRAY2BGR), 0.3, 0)  # 调整混合权重

    # 保存结果
    cv2.imwrite(output_path, mixed_image)

# 图片路径
image_path = "D:\\way\\yolo\\Datasets\\images_Rainy\\test\\berlin\\berlin_000000_000019_leftImg8bit.png"

# 输出图片路径
output_path = "D:\\way\\yolo\\000p.png"

# 使用示例
add_raindrop_effect(image_path, output_path)