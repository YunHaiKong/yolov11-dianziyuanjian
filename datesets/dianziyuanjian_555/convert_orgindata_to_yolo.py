import os
import json
import shutil
import random
from pathlib import Path
import math

def get_class_mapping():
    """
    获取类别映射
    """
    class_names = ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']
    return {name: idx for idx, name in enumerate(class_names)}

def rotated_rect_to_yolo(points, img_width, img_height):
    """
    将旋转矩形的点转换为YOLO格式的边界框
    points: 旋转矩形的8个点坐标
    """
    # 获取所有x和y坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    # 计算边界框
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # 转换为YOLO格式 (中心点坐标和宽高，都是相对值)
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return center_x, center_y, width, height

def convert_json_to_yolo(json_path, class_mapping):
    """
    将JSON标注文件转换为YOLO格式
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    yolo_annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label in class_mapping:
            class_id = class_mapping[label]
            points = shape['points']
            
            # 转换为YOLO格式
            center_x, center_y, width, height = rotated_rect_to_yolo(points, img_width, img_height)
            
            # 格式化为YOLO标注行
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def filter_duplicate_files(source_dir):
    """
    过滤掉带有(1)的重复文件，只保留原始文件
    """
    image_files = []
    json_files = []
    
    for file in os.listdir(source_dir):
        # 跳过带有(1)的文件
        if '(1)' in file:
            continue
            
        file_path = os.path.join(source_dir, file)
        if file.endswith(('.bmp', '.jpeg', '.jpg', '.png')):
            # 检查对应的JSON文件是否存在
            json_file = file + '.json'
            json_path = os.path.join(source_dir, json_file)
            if os.path.exists(json_path):
                image_files.append(file_path)
                json_files.append(json_path)
    
    return list(zip(image_files, json_files))

def create_yolo_dataset(source_dir, output_dir, train_ratio=0.8):
    """
    创建YOLO格式的数据集
    """
    # 创建输出目录
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取类别映射
    class_mapping = get_class_mapping()
    
    # 过滤重复文件
    file_pairs = filter_duplicate_files(source_dir)
    print(f"找到 {len(file_pairs)} 个有效的图像-标注对")
    
    # 随机打乱文件列表
    random.shuffle(file_pairs)
    
    # 计算分割点
    split_point = int(len(file_pairs) * train_ratio)
    train_pairs = file_pairs[:split_point]
    val_pairs = file_pairs[split_point:]
    
    print(f"训练集: {len(train_pairs)} 个文件")
    print(f"验证集: {len(val_pairs)} 个文件")
    
    # 处理训练集
    for img_path, json_path in train_pairs:
        # 复制图像文件
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(train_images_dir, img_name))
        
        # 转换标注文件
        yolo_annotations = convert_json_to_yolo(json_path, class_mapping)
        
        # 保存YOLO标注文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_annotations))
    
    # 处理验证集
    for img_path, json_path in val_pairs:
        # 复制图像文件
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(val_images_dir, img_name))
        
        # 转换标注文件
        yolo_annotations = convert_json_to_yolo(json_path, class_mapping)
        
        # 保存YOLO标注文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(val_labels_dir, label_name)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_annotations))
    
    # 创建dataset.yaml文件
    dataset_yaml = f"""path: c:\\Users\\Administrator\\Desktop\\11
train: train
val: val
nc: 5
names: ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']
"""
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(dataset_yaml)
    
    # 创建classes.txt文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for class_name in ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']:
            f.write(class_name + '\n')
    
    print("\n数据集转换完成！")
    print(f"训练集图像: {len(train_pairs)}")
    print(f"验证集图像: {len(val_pairs)}")
    print(f"总计: {len(file_pairs)} 个图像")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 源数据目录和输出目录
    source_directory = "data"
    output_directory = "yolo_dataset"
    
    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 源目录 {source_directory} 不存在")
        exit(1)
    
    # 创建YOLO数据集
    create_yolo_dataset(source_directory, output_directory, train_ratio=0.8)
    
    print("\n转换完成！现在可以运行 python train_yolo.py 开始训练。")