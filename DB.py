import os
import json

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

json_file = 'D:\way\yolo\Datasets\COCO2017\labels\instances_train2017.json'  # 你的JSON文件路径
save_path = 'D:\way\yolo\Datasets\COCO2017\labels\\train'  # 保存YOLO标签的路径

data = json.load(open(json_file, 'r'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

for img in data['images']:
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    ana_txt_name = head + ".txt"
    f_txt = open(os.path.join(save_path, ana_txt_name), 'w')
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            box = convert((img_width, img_height), ann["bbox"])
            # YOLO格式要求的类别标签通常是从0开始的索引，这里需要根据COCO数据集中的类别ID映射到YOLO的类别索引
            f_txt.write("%s %s %s %s %s\n" % (ann["category_id"], box[0], box[1], box[2], box[3]))
    f_txt.close()