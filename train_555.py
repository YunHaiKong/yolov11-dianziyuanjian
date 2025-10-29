
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib

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

def train_yolo():
    """训练YOLO模型"""
    
    # 加载预训练模型
    model = YOLO('yolo11n.pt')  # 使用nano版本，训练速度快
    
    # 开始训练
    results = model.train(
        data='ultralytics\cfg\datasets\dianziyuanjian_555.yaml',  # 数据集配置文件
        epochs=100,            # 训练轮数
        imgsz=640,           # 图片尺寸1
        batch=8,             # 批次大小2
        device='0',        # 使用GPU训练
        workers=2,           # 数据加载线程数
        patience=10,         # 早停耐心值1
        save=True,           # 保存模型
        plots=True,          # 生成训练图表
        val=True,           # 暂时关闭验证
        project='runs/dianziyuanjian_555', # 保存目录
        name='train', # 实验名称
    )
    
    print("训练完成!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    
    return results

def validate_model(model_path='runs/train/dianziyuanjian_555/weights/best.pt'):
    """验证模型"""
    if os.path.exists(model_path):
        model = YOLO(model_path)
        results = model.val(data='dataset.yaml')
        print(f"验证结果: mAP50={results.box.map50:.3f}, mAP50-95={results.box.map:.3f}")
        return results
    else:
        print(f"模型文件不存在: {model_path}")
        return None

def predict_sample(model_path='runs/train/dianziyuanjian_555/weights/best.pt', image_path='images'):
    """使用训练好的模型进行预测"""
    if os.path.exists(model_path):
        model = YOLO(model_path)
        
        # 获取第一张图片进行测试
        import glob
        image_files = glob.glob(os.path.join(image_path, '*'))
        if image_files:
            test_image = image_files[0]
            results = model(test_image)
            
            # 保存预测结果
            results[0].save('prediction_result.jpg')
            print(f"预测完成，结果保存为: prediction_result.jpg")
            
            # 打印检测结果
            for r in results:
                print(f"检测到 {len(r.boxes)} 个目标")
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_names = ['聚酯电容', '热敏电阻', '三极管损坏', '三极管完好', '三极管弯曲']
                        print(f"  - {class_names[class_id]}: {confidence:.3f}")
        else:
            print("没有找到图片文件")
    else:
        print(f"模型文件不存在: {model_path}")

if __name__ == '__main__':
    print("YOLO电子元件检测训练脚本")
    print("1. 开始训练")
    print("2. 验证模型")
    print("3. 预测测试")
    
    choice = input("请选择操作 (1/2/3): ")
    
    if choice == '1':
        train_yolo()
    elif choice == '2':
        validate_model()
    elif choice == '3':
        predict_sample()
    else:
        print("无效选择")
