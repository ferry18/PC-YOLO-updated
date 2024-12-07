import os
import numpy as np
import cv2


# beta:雾气强度，brightness：雾霾的亮度，决定了雾的明亮程度。
def add_hazy(image, beta=0.01, brightness=0.2):
    '''
    :param image:   输入图像
    :param beta:    雾强
    :param brightness:  雾霾亮度
    :return:    雾图
    '''
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))
    center = (row // 2, col // 2)
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return hazy_img

def process_directory(input_dir, output_dir, beta=0.05, brightness=0.5):
    '''
    处理指定目录下的所有图片并将结果保存到相同目录结构
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param beta: 雾强
    :param brightness: 雾霾亮度
    '''
    # 遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 读取图片
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)

                # 添加雾效
                image_fog = add_hazy(image, beta, brightness)

                # 获取相对路径
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                # 如果输出目录不存在，创建目录
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # 保存带雾的图片
                output_file = os.path.join(output_path, file)
                cv2.imwrite(output_file, image_fog)
                print(f"Processed and saved: {output_file}")

if __name__ == '__main__':
    # 设置输入和输出目录
    input_dir = r"D:\way\text"  # 修改为你自己的输入目录路径
    output_dir = r"D:\way\text4"  # 修改为你希望输出的目录路径

    # 调用函数处理目录中的图片   beta:雾气强度，brightness：雾霾的亮度，决定了雾的明亮程度。
    process_directory(input_dir, output_dir, beta=0.07, brightness=0.5)