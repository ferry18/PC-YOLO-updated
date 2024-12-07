import os

'''
图片计数器
'''
def count_images(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """
    递归统计指定目录下所有图片的数量。

    :param directory: 要统计的目录路径。
    :param extensions: 支持的图片文件扩展名列表。
    :return: 图片总数。
    """
    image_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_count += 1
                print(f"Found image: {os.path.join(root, file)}")  # 打印找到的图片路径
    return image_count

# 使用示例
if __name__ == "__main__":
    directory_to_check = r"D:\way\yolo\Datasets\RTTS\images\val"  # 指定的目录路径
    image_count = count_images(directory_to_check)
    print(f"\n在目录 {directory_to_check} 中找到 {image_count} 张图片。")