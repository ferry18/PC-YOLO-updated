import os
import numpy as np
import cv2

'''
雾气+雨滴在摄像头上

①Python 3.6 或更高版本
②需要安装opencv-python和numpy
pip install opencv-python numpy
'''


# 添加雨滴效果的函数
# brightness_factor: 雨滴的亮度因子，决定雨滴的透明度
# min_drop_size: 最小雨滴尺寸，以像素为单位
# max_drop_size: 最大雨滴尺寸，以像素为单位
def add_raindrop_effect(image, brightness_factor=0.7, min_drop_size=100, max_drop_size=600):
    if image is None:
        print("Error: Image is None.")
        return None

    width, height = image.shape[1], image.shape[0]  # 获取图像的宽度和高度
    raindrop_mask = np.zeros((height, width), dtype=np.uint8)  # 创建雨滴遮罩层

    # 随机生成雨滴数量，范围在10到16之间
    num_drops = np.random.randint(9, 14)
    for _ in range(num_drops):
        x = np.random.randint(0, width)  # 随机生成雨滴的x坐标
        y = np.random.randint(0, height)  # 随机生成雨滴的y坐标
        drop_size = np.random.randint(min_drop_size, max_drop_size)  # 随机生成雨滴的大小
        for i in range(-drop_size // 2, drop_size // 2 + 1):
            for j in range(-drop_size // 2, drop_size // 2 + 1):
                if 0 <= y + i < height and 0 <= x + j < width:  # 确保不超出图像边界
                    distance = (i ** 2 + j ** 2) ** 0.5  # 计算雨滴中心到当前点的距离
                    if distance <= drop_size // 2:  # 如果距离小于雨滴半径
                        opacity = int((1 - (distance / (drop_size // 2))) * 255 * brightness_factor)  # 计算透明度
                        raindrop_mask[y + i, x + j] = opacity  # 应用透明度到遮罩层

    mixed_image = cv2.addWeighted(image, 1, cv2.cvtColor(raindrop_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)  # 将雨滴遮罩与原图混合
    return mixed_image

# 添加雾化效果的函数
# beta: 雾的强度参数，影响雾的浓度
# brightness: 雾的亮度，决定雾的明亮程度
def add_hazy(image, beta=0.001, brightness=0.2):
    img_f = image.astype(np.float32) / 255.0  # 将图像转换为浮点数并归一化到[0, 1]
    row, col, chs = image.shape  # 获取图像的尺寸
    size = np.sqrt(max(row, col))  # 计算图像对角线长度
    center = (row // 2, col // 2)  # 获取图像中心点坐标
    y, x = np.ogrid[:row, :col]  # 创建坐标网格
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)  # 计算每个点到中心的距离
    d = -0.04 * dist + size  # 计算雾化因子
    td = np.exp(-beta * d)  # 应用雾化因子
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])  # 应用雾化效果
    hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)  # 将结果转换回[0, 255]范围的整数
    return hazy_img

# 处理目录中所有图片的函数
def process_directory(input_dir, output_dir, beta=0.05, brightness=0.5, raindrop_brightness=0.7, raindrop_min_size=20, raindrop_max_size=600):
    for root, dirs, files in os.walk(input_dir):  # 遍历输入目录
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
                file_path = os.path.join(root, file)  # 获取文件完整路径
                image = cv2.imread(file_path)  # 读取图像
                if image is None:  # 如果图像读取失败
                    print(f"Failed to read image: {file_path}")
                    continue

                # 添加雾化效果
                image_fog = add_hazy(image, beta, brightness)
                # 添加雨滴效果
                image_rain = add_raindrop_effect(image_fog, raindrop_brightness, raindrop_min_size, raindrop_max_size)
                if image_rain is None:  # 如果雨滴效果添加失败
                    continue

                # 获取输出目录的相对路径
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)  # 构建输出目录路径

                # 如果输出目录不存在，则创建目录
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # 保存处理后的图像
                output_file = os.path.join(output_path, file)
                cv2.imwrite(output_file, image_rain)  # 写入图像到文件
                print(f"Processed and saved: {output_file}")  # 打印处理结果

if __name__ == '__main__':
    input_dir = r"D:\way\text"  # 输入目录路径
    output_dir = r"D:\way\text4"  # 输出目录路径

    # 调用处理函数
    # input_dir: 输入目录路径
    # output_dir: 输出目录路径
    # beta: 雾强度参数
    # brightness: 雾亮度参数
    # raindrop_brightness: 雨滴亮度因子
    # raindrop_min_size: 最小雨滴尺寸
    # raindrop_max_size: 最大雨滴尺寸
    '''
    其中raindrop_min_size最小雨滴尺寸、raindrop_max_size最大雨滴尺寸要根据图片大小适当调整，其他参数也可以
    '''
    process_directory(input_dir, output_dir, beta=0.03, brightness=0.5, raindrop_brightness=0.7, raindrop_min_size=100, raindrop_max_size=500)