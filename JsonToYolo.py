import json
import os

# 自动生成类别映射表
class_mapping = {}

def convert_cityscapes_to_yolo(json_file, output_dir, img_width, img_height):
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded JSON data from {json_file}: {data}")  # 打印数据

    if 'objects' not in data or not data['objects']:
        print(f"No objects found in {json_file}. Skipping...")
        return  # 跳过该文件

    yolo_labels = []
    for obj in data['objects']:
        if 'polygon' not in obj:
            print(f"No polygon found for object in {json_file}. Skipping...")
            continue  # 跳过没有 polygon 的对象

        label = obj['label']

        # 计算边界框
        polygon = obj['polygon']
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)

        # 如果类别不在映射表中，为其分配一个唯一的ID
        if label not in class_mapping:
            class_mapping[label] = len(class_mapping)

        class_id = class_mapping[label]

        # 计算 YOLO 中心点和宽高，归一化处理
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 生成与原始 JSON 文件相同的目录结构
    relative_path = os.path.relpath(json_file, start=json_dir)
    output_file_dir = os.path.join(output_dir, os.path.dirname(relative_path))

    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    # 将 YOLO 标签保存到文件中
    output_file = os.path.join(output_file_dir, f"{os.path.basename(json_file).split('.')[0]}.txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_labels))
    print(f"Saved YOLO labels to {output_file}")

# 保存类别映射表到文件
def save_class_mapping(output_dir):
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for label, class_id in class_mapping.items():
            f.write(f"{class_id}: {label}\n")

# 递归查找 JSON 文件
def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

# 使用示例
def convert_all_json_to_yolo(json_dir, output_dir, img_width, img_height):
    json_files = find_json_files(json_dir)  # 查找所有 JSON 文件
    for json_file in json_files:
        print(f"Processing file: {json_file}")  # 添加打印语句
        convert_cityscapes_to_yolo(json_file, output_dir, img_width, img_height)

    save_class_mapping(output_dir)

# 示例调用
json_dir = 'D:\\way\\yolo\\Datasets\\cityscapes\\gtFine'  # JSON 文件所在目录
output_dir = 'D:\\way\\yolo\\Datasets\\cityscapes\\gtFine-label'  # 输出标签文件所在目录
img_width = 2048  # 图像宽度
img_height = 1024  # 图像高度

convert_all_json_to_yolo(json_dir, output_dir, img_width, img_height)


