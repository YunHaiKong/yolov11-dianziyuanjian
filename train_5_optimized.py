import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import torch
import yaml

# 设置中文字体支持，解决训练时的字体警告
def setup_chinese_font():
    """设置matplotlib支持中文字体显示"""
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'DejaVu Sans'       # 备用字体
    ]
    
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"设置中文字体: {font_name}")
            break
        except:
            continue
    
    # 清除matplotlib字体缓存
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass

# 在导入后立即设置字体
setup_chinese_font()

def analyze_dataset(data_yaml_path):
    """分析数据集类别分布"""
    print("\n=== 数据集分析 ===")
    
    # 读取数据集配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    train_path = os.path.join(data_config['path'], data_config['train'], 'labels')
    val_path = os.path.join(data_config['path'], data_config['val'], 'labels')
    
    # 统计各类别样本数量
    class_counts = {i: 0 for i in range(data_config['nc'])}
    
    # 统计训练集
    if os.path.exists(train_path):
        for label_file in os.listdir(train_path):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_path, label_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
    
    print("训练集类别分布:")
    for i, count in class_counts.items():
        class_name = data_config['names'][i]
        print(f"  {class_name}: {count} 个样本")
    
    # 计算类别权重（用于处理类别不平衡）
    total_samples = sum(class_counts.values())
    class_weights = {}
    for i, count in class_counts.items():
        if count > 0:
            class_weights[i] = total_samples / (len(class_counts) * count)
        else:
            class_weights[i] = 1.0
    
    print("\n建议的类别权重:")
    for i, weight in class_weights.items():
        class_name = data_config['names'][i]
        print(f"  {class_name}: {weight:.2f}")
    
    return class_weights

def train_yolo_optimized():
    """优化版YOLO训练"""
    print("\n=== 开始优化训练 ===")
    
    # 分析数据集
    data_yaml = 'ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml'
    class_weights = analyze_dataset(data_yaml)
    
    # 使用更大的模型以提高性能
    model = YOLO('yolo11s.pt')  # 从nano升级到small版本
    
    # 优化的训练参数
    results = model.train(
        data=data_yaml,
        epochs=150,              # 增加训练轮数
        imgsz=640,              # 保持图片尺寸
        batch=6,                # 适当减小批次大小以适应更大模型
        device='0',             # 使用GPU训练
        workers=4,              # 增加数据加载线程数
        patience=20,            # 增加早停耐心值
        save=True,              # 保存模型
        plots=True,             # 生成训练图表
        val=True,               # 启用验证
        project='runs/Dianziyuanjian_5_optimized',
        name='train',
        
        # 优化参数
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率因子
        momentum=0.937,         # SGD动量
        weight_decay=0.0005,    # 权重衰减
        warmup_epochs=3,        # 预热轮数
        warmup_momentum=0.8,    # 预热动量
        warmup_bias_lr=0.1,     # 预热偏置学习率
        
        # 数据增强参数（针对小目标和难检测类别）
        hsv_h=0.015,            # 色调增强
        hsv_s=0.7,              # 饱和度增强
        hsv_v=0.4,              # 明度增强
        degrees=10.0,           # 旋转角度
        translate=0.1,          # 平移
        scale=0.5,              # 缩放
        shear=2.0,              # 剪切
        perspective=0.0001,     # 透视变换
        flipud=0.0,             # 上下翻转概率
        fliplr=0.5,             # 左右翻转概率
        mosaic=1.0,             # Mosaic增强概率
        mixup=0.1,              # Mixup增强概率
        copy_paste=0.1,         # Copy-paste增强概率
        
        # 损失函数权重
        box=7.5,                # 边界框损失权重
        cls=0.5,                # 分类损失权重
        dfl=1.5,                # 分布焦点损失权重
        
        # 其他优化参数
        close_mosaic=10,        # 最后10轮关闭mosaic
        amp=True,               # 自动混合精度
        fraction=1.0,           # 使用全部数据
        profile=False,          # 关闭性能分析以节省时间
        freeze=None,            # 不冻结任何层
        
        # 验证参数
        save_period=10,         # 每10轮保存一次
    )
    
    print("\n=== 训练完成 ===")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最新模型保存在: {results.save_dir}/weights/last.pt")
    
    return results

def train_yolo_with_class_weights():
    """使用类别权重的训练方法（实验性）"""
    print("\n=== 开始类别权重训练 ===")
    
    # 使用中等大小的模型
    model = YOLO('yolo11m.pt')  # medium版本，平衡性能和速度
    
    # 基础训练参数
    results = model.train(
        data='ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml',
        epochs=200,              # 更多训练轮数
        imgsz=640,
        batch=4,                # 更小批次以适应medium模型
        device='0',
        workers=4,
        patience=30,            # 更大的耐心值
        save=True,
        plots=True,
        val=True,
        project='runs/Dianziyuanjian_5_weighted',
        name='train',
        
        # 针对类别不平衡的优化
        lr0=0.008,              # 稍低的学习率
        lrf=0.005,
        momentum=0.9,
        weight_decay=0.001,     # 更强的正则化
        
        # 更强的数据增强
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=15.0,           # 更大的旋转角度
        translate=0.15,         # 更大的平移
        scale=0.6,              # 更大的缩放范围
        shear=3.0,              # 更大的剪切
        perspective=0.0002,
        flipud=0.1,             # 增加上下翻转
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,             # 增加mixup
        copy_paste=0.15,        # 增加copy-paste
        
        # 调整损失权重
        box=8.0,                # 增加边界框损失权重
        cls=1.0,                # 增加分类损失权重
        dfl=2.0,                # 增加分布焦点损失权重
        
        close_mosaic=15,        # 最后15轮关闭mosaic
        amp=True,
        save_period=5,          # 更频繁保存
    )
    
    print("\n=== 类别权重训练完成 ===")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    
    return results

def validate_model(model_path):
    """验证模型性能"""
    if os.path.exists(model_path):
        print(f"\n=== 验证模型: {model_path} ===")
        model = YOLO(model_path)
        results = model.val(data='ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml')
        
        print(f"\n验证结果:")
        print(f"  总体 mAP50: {results.box.map50:.3f}")
        print(f"  总体 mAP50-95: {results.box.map:.3f}")
        
        # 显示各类别性能
        class_names = ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']
        if hasattr(results.box, 'maps'):
            print("\n各类别性能:")
            for i, (name, map_val) in enumerate(zip(class_names, results.box.maps)):
                print(f"  {name}: mAP50-95 = {map_val:.3f}")
        
        return results
    else:
        print(f"模型文件不存在: {model_path}")
        return None

def predict_sample(model_path, image_path='datesets/dianziyuanjian_5/val/images'):
    """使用训练好的模型进行预测"""
    if os.path.exists(model_path):
        print(f"\n=== 使用模型预测: {model_path} ===")
        model = YOLO(model_path)
        
        # 获取测试图片
        import glob
        image_files = glob.glob(os.path.join(image_path, '*'))
        if image_files:
            test_image = image_files[0]
            print(f"测试图片: {test_image}")
            
            results = model(test_image, conf=0.25, iou=0.45)
            
            # 保存预测结果
            output_path = 'prediction_result_optimized.jpg'
            results[0].save(output_path)
            print(f"预测结果保存为: {output_path}")
            
            # 打印检测结果
            class_names = ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']
            for r in results:
                print(f"\n检测到 {len(r.boxes)} 个目标:")
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        print(f"  - {class_names[class_id]}: {confidence:.3f}")
                else:
                    print("  未检测到任何目标")
        else:
            print(f"在 {image_path} 中没有找到图片文件")
    else:
        print(f"模型文件不存在: {model_path}")

def compare_models():
    """比较不同训练方法的模型性能"""
    print("\n=== 模型性能比较 ===")
    
    models_to_compare = [
        ('原始模型', 'runs/Dianziyuanjian_5/train/weights/best.pt'),
        ('优化模型', 'runs/Dianziyuanjian_5_optimized/train/weights/best.pt'),
        ('权重模型', 'runs/Dianziyuanjian_5_weighted/train/weights/best.pt'),
    ]
    
    results_summary = []
    
    for name, path in models_to_compare:
        if os.path.exists(path):
            print(f"\n--- {name} ---")
            model = YOLO(path)
            results = model.val(data='ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml')
            
            summary = {
                'name': name,
                'map50': results.box.map50,
                'map50_95': results.box.map,
                'path': path
            }
            results_summary.append(summary)
            
            print(f"mAP50: {results.box.map50:.3f}, mAP50-95: {results.box.map:.3f}")
        else:
            print(f"模型文件不存在: {path}")
    
    # 找出最佳模型
    if results_summary:
        best_model = max(results_summary, key=lambda x: x['map50_95'])
        print(f"\n=== 最佳模型 ===")
        print(f"模型: {best_model['name']}")
        print(f"mAP50: {best_model['map50']:.3f}")
        print(f"mAP50-95: {best_model['map50_95']:.3f}")
        print(f"路径: {best_model['path']}")

if __name__ == '__main__':
    print("YOLO电子元件检测优化训练脚本")
    print("1. 数据集分析")
    print("2. 优化训练 (推荐)")
    print("3. 类别权重训练")
    print("4. 验证模型")
    print("5. 预测测试")
    print("6. 模型比较")
    print("7. 全流程优化")
    
    choice = input("\n请选择操作 (1-7): ")
    
    if choice == '1':
        analyze_dataset('ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml')
    elif choice == '2':
        train_yolo_optimized()
    elif choice == '3':
        train_yolo_with_class_weights()
    elif choice == '4':
        model_path = input("请输入模型路径: ")
        validate_model(model_path)
    elif choice == '5':
        model_path = input("请输入模型路径: ")
        predict_sample(model_path)
    elif choice == '6':
        compare_models()
    elif choice == '7':
        print("\n=== 开始全流程优化 ===")
        # 1. 数据集分析
        analyze_dataset('ultralytics\\cfg\\datasets\\dianziyuanjian_5.yaml')
        
        # 2. 优化训练
        print("\n开始优化训练...")
        train_yolo_optimized()
        
        # 3. 验证结果
        print("\n验证优化模型...")
        validate_model('runs/Dianziyuanjian_5_optimized/train/weights/best.pt')
        
        # 4. 预测测试
        print("\n测试预测效果...")
        predict_sample('runs/Dianziyuanjian_5_optimized/train/weights/best.pt')
        
        print("\n=== 全流程优化完成 ===")
    else:
        print("无效选择")