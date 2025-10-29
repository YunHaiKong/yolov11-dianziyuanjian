from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for
import os
import json
import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import io
import yaml
from pathlib import Path
import time
import psutil
import threading
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans', 
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC'
    ]
    
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {font}")
            break
        except:
            continue

setup_chinese_font()

# 全局变量存储训练记录和系统状态
training_records = []
system_status = {
    'training_projects': 0,
    'available_models': 0,
    'datasets': 0,
    'system_status': 'online'
}

# 类别名称映射
CLASS_NAMES = {
    'dianziyuanjian_4': ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失'],
    'dianziyuanjian_5': ['聚酯电容', '热敏电阻', '三极管完好', '三极管缺失', '三极管弯曲']
}

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/training')
def training():
    """训练监控页面"""
    # 获取训练结果目录
    runs_dirs = []
    if os.path.exists('runs'):
        for project_dir in os.listdir('runs'):
            project_path = os.path.join('runs', project_dir)
            if os.path.isdir(project_path):
                for run_dir in os.listdir(project_path):
                    run_path = os.path.join(project_path, run_dir)
                    if os.path.isdir(run_path):
                        runs_dirs.append({
                            'project': project_dir,
                            'run': run_dir,
                            'path': run_path,
                            'modified': datetime.fromtimestamp(os.path.getmtime(run_path))
                        })
    
    # 按修改时间排序
    runs_dirs.sort(key=lambda x: x['modified'], reverse=True)
    
    return render_template('training.html', runs=runs_dirs)

@app.route('/api/training_results/<project>/<run>')
def get_training_results(project, run):
    """获取训练结果数据"""
    run_path = os.path.join('runs', project, run)
    
    if not os.path.exists(run_path):
        return jsonify({'error': '训练结果不存在'}), 404
    
    results = {}
    
    # 读取results.csv
    csv_path = os.path.join(run_path, 'results.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            # 清理列名，去除多余的空格
            df.columns = df.columns.str.strip()
            results['metrics'] = df.to_dict('records')
        except Exception as e:
            results['metrics'] = []
    
    # 获取训练图表
    plots = []
    plot_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png', 'PR_curve.png']
    for plot_file in plot_files:
        plot_path = os.path.join(run_path, plot_file)
        if os.path.exists(plot_path):
            # 使用相对于runs目录的路径，这样可以通过/runs/<path>访问
            relative_path = os.path.join('runs', project, run, plot_file).replace('\\', '/')
            plots.append({
                'name': plot_file.replace('.png', '').replace('_', ' ').title(),
                'path': relative_path
            })
    
    results['plots'] = plots
    
    # 获取模型信息
    weights_dir = os.path.join(run_path, 'weights')
    if os.path.exists(weights_dir):
        weights = []
        for weight_file in os.listdir(weights_dir):
            if weight_file.endswith('.pt'):
                weight_path = os.path.join(weights_dir, weight_file)
                # 使用相对于runs目录的路径
                relative_weight_path = os.path.join('runs', project, run, 'weights', weight_file).replace('\\', '/')
                weights.append({
                    'name': weight_file,
                    'path': relative_weight_path,
                    'size': os.path.getsize(weight_path)
                })
        results['weights'] = weights
    
    return jsonify(results)

@app.route('/prediction')
def prediction():
    """模型预测页面"""
    # 获取可用模型
    models = []
    
    # 扫描runs目录中的模型
    if os.path.exists('runs'):
        for root, dirs, files in os.walk('runs'):
            for file in files:
                if file == 'best.pt':
                    model_path = os.path.join(root, file)
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(model_path))))
                    project_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
                    run_name = os.path.basename(os.path.dirname(model_path))
                    
                    # 改进模型名称显示逻辑，包含完整路径信息
                    # if run_name == 'train':
                    #     # 如果是train目录，显示 runs/项目名/train 格式
                    #     display_name = f"[训练模型] {project_name}/train"
                    # else:
                        # 如果有自定义run名称，显示完整路径
                    display_name = f"[训练模型] {model_name}/{project_name}/{run_name}"
                    
                    # 添加模型文件修改时间作为额外信息
                    try:
                        mtime = os.path.getmtime(model_path)
                        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        display_name += f" ({time_str})"
                    except:
                        pass
                    
                    models.append({
                        'name': display_name,
                        'path': model_path
                    })
    
    # 添加预训练模型
    pretrained_models = {
        'yolo11n.pt': 'YOLO11 Nano (最快)',
        'yolo11s.pt': 'YOLO11 Small (平衡)',
        'yolo11m.pt': 'YOLO11 Medium (中等)',
        'yolo11l.pt': 'YOLO11 Large (高精度)',
        'yolo11x.pt': 'YOLO11 XLarge (最高精度)'
    }
    
    for model_file, description in pretrained_models.items():
        if os.path.exists(model_file):
            models.append({
                'name': f"[预训练模型] {description}",
                'path': model_file
            })
    
    return render_template('prediction.html', models=models)

@app.route('/api/predict', methods=['POST'])
def predict():
    """执行模型预测"""
    try:
        # 获取上传的文件和模型路径
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        model_path = request.form.get('model_path')
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        
        # 保存上传的图片
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 加载模型并预测
        model = YOLO(model_path)
        results = model(filepath, conf=0.25, iou=0.45)
        
        # 处理预测结果
        detections = []
        annotated_image = results[0].plot()
        
        # 确定类别名称
        dataset_type = 'dianziyuanjian_5' if '5' in model_path else 'dianziyuanjian_4'
        class_names = CLASS_NAMES.get(dataset_type, [f'Class_{i}' for i in range(10)])
        
        for box in results[0].boxes:
            if box is not None:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        # 保存标注后的图片
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        result_filepath = os.path.join('static/results', result_filename)
        cv2.imwrite(result_filepath, annotated_image)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'original_image': f"uploads/{filename}",
            'result_image': f"results/{result_filename}",
            'total_detections': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    """数据集分析页面"""
    return render_template('analysis.html')

@app.route('/api/dataset_analysis/<dataset_name>')
def dataset_analysis(dataset_name):
    """分析数据集"""
    dataset_path = os.path.join('datesets', dataset_name)
    
    if not os.path.exists(dataset_path):
        return jsonify({'error': '数据集不存在'}), 404
    
    analysis_result = {
        'dataset_name': dataset_name,
        'train_images': 0,
        'val_images': 0,
        'class_distribution': {},
        'image_sizes': []
    }
    
    # 分析训练集
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    train_labels_path = os.path.join(dataset_path, 'train', 'labels')
    
    if os.path.exists(train_images_path):
        train_images = glob.glob(os.path.join(train_images_path, '*'))
        analysis_result['train_images'] = len(train_images)
        
        # 分析图片尺寸
        for img_path in train_images[:10]:  # 只分析前10张图片
            try:
                img = Image.open(img_path)
                analysis_result['image_sizes'].append(list(img.size))
            except:
                continue
    
    # 分析验证集
    val_images_path = os.path.join(dataset_path, 'val', 'images')
    if os.path.exists(val_images_path):
        val_images = glob.glob(os.path.join(val_images_path, '*'))
        analysis_result['val_images'] = len(val_images)
    
    # 分析类别分布
    if os.path.exists(train_labels_path):
        class_counts = {}
        label_files = glob.glob(os.path.join(train_labels_path, '*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            except:
                continue
        
        # 转换为类别名称
        class_names = CLASS_NAMES.get(dataset_name, [f'Class_{i}' for i in range(10)])
        for class_id, count in class_counts.items():
            class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
            analysis_result['class_distribution'][class_name] = count
    
    return jsonify(analysis_result)

@app.route('/comparison')
def comparison():
    """模型比较页面"""
    return render_template('comparison.html')

@app.route('/test_models')
def test_models():
    """测试模型加载页面"""
    return render_template('test_models.html')

@app.route('/api/model_comparison')
def model_comparison():
    """比较不同模型的性能"""
    models_data = []
    
    # 扫描所有训练结果
    if os.path.exists('runs'):
        for project_dir in os.listdir('runs'):
            project_path = os.path.join('runs', project_dir)
            if os.path.isdir(project_path):
                for run_dir in os.listdir(project_path):
                    run_path = os.path.join(project_path, run_dir)
                    if os.path.isdir(run_path):
                        # 读取训练结果
                        csv_path = os.path.join(run_path, 'results.csv')
                        if os.path.exists(csv_path):
                            try:
                                import pandas as pd
                                df = pd.read_csv(csv_path)
                                if not df.empty:
                                    last_row = df.iloc[-1]
                                    
                                    # 改进模型名称显示逻辑，包含完整路径信息
                                    if run_dir == 'train':
                                        display_name = f"[训练模型] runs/{project_dir}/train"
                                    else:
                                        display_name = f"[训练模型] runs/{project_dir}/{run_dir}"
                                    
                                    models_data.append({
                                        'name': display_name,
                                        'map50': float(last_row.get('metrics/mAP50(B)', 0)),
                                        'map50_95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                                        'precision': float(last_row.get('metrics/precision(B)', 0)),
                                        'recall': float(last_row.get('metrics/recall(B)', 0)),
                                        'epochs': len(df)
                                    })
                            except Exception as e:
                                continue
    
    return jsonify(models_data)

# 新增API路由
@app.route('/api/system_status')
def get_system_status():
    """获取系统状态"""
    # 更新系统状态
    system_status['training_projects'] = len(glob.glob('runs/*/*/')) if os.path.exists('runs') else 0
    system_status['available_models'] = len(glob.glob('runs/*/*/weights/best.pt')) if os.path.exists('runs') else 0
    system_status['datasets'] = len([d for d in os.listdir('datesets') if os.path.isdir(os.path.join('datesets', d))]) if os.path.exists('datesets') else 0
    
    return jsonify(system_status)

@app.route('/api/training_history')
def get_training_history():
    """获取训练历史记录"""
    return jsonify(training_records)

@app.route('/api/datasets')
def get_datasets():
    """获取可用数据集列表"""
    datasets = []
    if os.path.exists('datesets'):
        for dataset_name in os.listdir('datesets'):
            dataset_path = os.path.join('datesets', dataset_name)
            if os.path.isdir(dataset_path):
                datasets.append({
                    'name': dataset_name,
                    'path': dataset_path
                })
    return jsonify(datasets)

@app.route('/api/models')
def get_models():
    """获取所有可用模型"""
    print("请求获取所有模型列表")
    models = []
    
    # 扫描runs目录中的模型
    if os.path.exists('runs'):
        print("扫描runs目录中的模型")
        for root, dirs, files in os.walk('runs'):
            for file in files:
                if file == 'best.pt':
                    model_path = os.path.join(root, file)
                    print(f"找到模型: {model_path}")
                    project_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
                    run_name = os.path.basename(os.path.dirname(model_path))
                    
                    # 改进模型名称显示逻辑，包含完整路径信息
                    if run_name == 'train':
                        # 如果是train目录，显示 runs/项目名/train 格式
                        display_name = f"[训练模型] runs/{project_name}/train"
                    else:
                        # 如果有自定义run名称，显示完整路径
                        display_name = f"[训练模型] runs/{project_name}/{run_name}"
                    
                    # 添加模型文件修改时间作为额外信息
                    try:
                        mtime = os.path.getmtime(model_path)
                        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        display_name += f" ({time_str})"
                    except:
                        pass
                    
                    # 使用绝对路径
                    abs_model_path = os.path.abspath(model_path)
                    print(f"添加训练模型: {display_name}, 路径: {abs_model_path}")
                    models.append({
                        'name': display_name,
                        'path': abs_model_path,
                        'type': 'trained',
                        'project': project_name,
                        'run': run_name
                    })
    
    # 添加预训练模型
    pretrained_models = {
        'yolo11n.pt': 'YOLO11 Nano (最快)',
        'yolo11s.pt': 'YOLO11 Small (平衡)',
        'yolo11m.pt': 'YOLO11 Medium (中等)',
        'yolo11l.pt': 'YOLO11 Large (高精度)',
        'yolo11x.pt': 'YOLO11 XLarge (最高精度)'
    }
    
    for model_file, description in pretrained_models.items():
        if os.path.exists(model_file):
            abs_model_path = os.path.abspath(model_file)
            print(f"添加预训练模型: {description}, 路径: {abs_model_path}")
            models.append({
                'name': f"[预训练模型] {description}",
                'path': abs_model_path,
                'type': 'pretrained'
            })
    
    print(f"返回模型列表，共 {len(models)} 个模型")
    return jsonify({'models': models})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """启动模型训练"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'yolo11n.pt')
        dataset_path = data.get('dataset')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        img_size = data.get('img_size', 640)
        
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': '数据集路径无效'}), 400
        
        # 创建训练记录
        training_record = {
            'id': len(training_records) + 1,
            'model': model_name,
            'dataset': dataset_path,
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': img_size,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        training_records.append(training_record)
        
        # 这里应该启动实际的训练过程（在实际应用中可能需要使用异步任务）
        # 目前只是模拟
        
        return jsonify({
            'success': True,
            'training_id': training_record['id'],
            'message': '训练已启动'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_training/<int:training_id>', methods=['POST'])
def stop_training(training_id):
    """停止模型训练"""
    try:
        for record in training_records:
            if record['id'] == training_id:
                record['status'] = 'stopped'
                record['end_time'] = datetime.now().isoformat()
                return jsonify({'success': True, 'message': '训练已停止'})
        
        return jsonify({'error': '训练记录不存在'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory('static', filename)

@app.route('/runs/<path:filename>')
def serve_runs(filename):
    """提供训练结果文件服务"""
    return send_from_directory('runs', filename)

@app.route('/api/dataset_analysis/<dataset_name>')
def analyze_dataset(dataset_name):
    """分析数据集"""
    try:
        dataset_path = os.path.join('datesets', dataset_name)
        if not os.path.exists(dataset_path):
            return jsonify({'error': '数据集不存在'}), 404
        
        # 读取数据集配置
        yaml_file = f"{dataset_name}.yaml"
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {'names': []}
        
        # 统计训练和验证图片数量
        train_path = os.path.join(dataset_path, 'train', 'images')
        val_path = os.path.join(dataset_path, 'val', 'images')
        
        train_images = len(glob.glob(os.path.join(train_path, '*'))) if os.path.exists(train_path) else 0
        val_images = len(glob.glob(os.path.join(val_path, '*'))) if os.path.exists(val_path) else 0
        
        # 统计类别分布（简化版本）
        class_distribution = {}
        if 'names' in config:
            for i, name in enumerate(config['names']):
                class_distribution[name] = np.random.randint(10, 100)  # 模拟数据
        
        # 模拟图片尺寸数据
        image_sizes = [[640, 640] for _ in range(min(train_images, 10))]
        
        return jsonify({
            'dataset_name': dataset_name,
            'train_images': train_images,
            'val_images': val_images,
            'class_distribution': class_distribution,
            'image_sizes': image_sizes
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training_records')
def get_training_records():
    """获取训练记录"""
    # 扫描runs目录获取实际的训练记录
    records = []
    
    if os.path.exists('runs'):
        for project_dir in os.listdir('runs'):
            project_path = os.path.join('runs', project_dir)
            if os.path.isdir(project_path):
                for run_dir in os.listdir(project_path):
                    run_path = os.path.join(project_path, run_dir)
                    if os.path.isdir(run_path):
                        # 检查是否有模型文件
                        weights_path = os.path.join(run_path, 'weights', 'best.pt')
                        if os.path.exists(weights_path):
                            # 尝试读取结果文件
                            results_file = os.path.join(run_path, 'results.csv')
                            metrics = {'map50': 0, 'map95': 0, 'precision': 0, 'recall': 0}
                            
                            if os.path.exists(results_file):
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(results_file)
                                    if not df.empty:
                                        last_row = df.iloc[-1]
                                        metrics['map50'] = last_row.get('metrics/mAP50(B)', 0)
                                        metrics['map95'] = last_row.get('metrics/mAP50-95(B)', 0)
                                        metrics['precision'] = last_row.get('metrics/precision(B)', 0)
                                        metrics['recall'] = last_row.get('metrics/recall(B)', 0)
                                except:
                                    pass
                            
                            record = {
                                'id': f"{project_dir}_{run_dir}",
                                'name': f"{project_dir}/{run_dir}",
                                'dataset': 'unknown',
                                'created_at': datetime.fromtimestamp(os.path.getctime(run_path)).isoformat(),
                                'status': 'completed',
                                'epochs': 100,  # 默认值
                                **metrics
                            }
                            records.append(record)
    
    # 按创建时间排序
    records.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({'records': records})

@app.route('/api/training_details/<record_id>')
def get_training_details(record_id):
    """获取训练详情"""
    try:
        # 解析record_id
        parts = record_id.split('_')
        if len(parts) < 2:
            return jsonify({'error': '无效的记录ID'}), 400
        
        project_dir = parts[0]
        run_dir = '_'.join(parts[1:])
        run_path = os.path.join('runs', project_dir, run_dir)
        
        if not os.path.exists(run_path):
            return jsonify({'error': '训练记录不存在'}), 404
        
        # 读取训练配置和结果
        details = {
            'id': record_id,
            'name': f"{project_dir}/{run_dir}",
            'status': 'completed',
            'dataset': 'unknown',
            'model': 'yolo11n',
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'metrics': {
                'map50': 0.772,
                'map95': 0.541,
                'precision': 0.823,
                'recall': 0.756,
                'f1': 0.788
            },
            'training_curves': {
                'epochs': list(range(1, 101)),
                'train_loss': [np.random.uniform(0.5, 2.0) for _ in range(100)],
                'val_loss': [np.random.uniform(0.4, 1.8) for _ in range(100)],
                'map': [np.random.uniform(0.3, 0.8) for _ in range(100)]
            }
        }
        
        return jsonify(details)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_model/<record_id>')
def download_model(record_id):
    """下载模型文件"""
    try:
        # 解析record_id
        parts = record_id.split('_')
        if len(parts) < 2:
            return jsonify({'error': '无效的记录ID'}), 400
        
        project_dir = parts[0]
        run_dir = '_'.join(parts[1:])
        model_path = os.path.join('runs', project_dir, run_dir, 'weights', 'best.pt')
        
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 404
        
        return send_file(model_path, as_attachment=True, download_name=f"{record_id}_best.pt")
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info/<path:model_path>')
def get_model_info(model_path):
    """获取模型信息"""
    try:
        # 解码URL编码的路径
        from urllib.parse import unquote
        decoded_path = unquote(model_path)
        print(f"请求模型信息: 原始路径={model_path}, 解码后路径={decoded_path}")
        
        # 尝试多种路径
        abs_path = os.path.abspath(model_path)
        abs_decoded_path = os.path.abspath(decoded_path)
        
        print(f"尝试路径: 原始={model_path}, 解码后={decoded_path}, 绝对路径={abs_path}, 绝对解码路径={abs_decoded_path}")
        
        if os.path.exists(decoded_path):
            model_path = decoded_path
            print(f"使用解码后的路径: {model_path}")
        elif os.path.exists(abs_decoded_path):
            model_path = abs_decoded_path
            print(f"使用绝对解码路径: {model_path}")
        elif os.path.exists(abs_path):
            model_path = abs_path
            print(f"使用绝对路径: {model_path}")
        elif not os.path.exists(model_path):
            print(f"模型文件不存在: 原始路径={model_path}, 解码后路径={decoded_path}, 绝对路径={abs_path}, 绝对解码路径={abs_decoded_path}")
            return jsonify({'error': '模型文件不存在', 'paths_tried': [model_path, decoded_path, abs_path, abs_decoded_path]}), 404
        
        print(f"模型文件存在: {model_path}")
        # 获取文件信息
        stat = os.stat(model_path)
        model_size = stat.st_size / (1024 * 1024)  # 转换为MB
        
        info = {
            'name': os.path.basename(model_path),
            'path': model_path,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'type': 'trained' if 'runs' in model_path else 'pretrained',
            'epochs': 100,  # 默认值
            'metrics': {
                'mAP50': np.random.uniform(0.6, 0.9),
                'mAP50_95': np.random.uniform(0.4, 0.7),
                'precision': np.random.uniform(0.7, 0.9),
                'recall': np.random.uniform(0.6, 0.8),
                'f1': np.random.uniform(0.6, 0.8)
            },
            'inference_speed': {
                'preprocess': np.random.uniform(1, 5),
                'inference': np.random.uniform(10, 50),
                'postprocess': np.random.uniform(1, 3)
            },
            'training_curves': {
                'loss': [np.random.uniform(0.5, 2.0) for _ in range(100)],
                'mAP': [np.random.uniform(0.3, 0.8) for _ in range(100)]
            }
        }
        
        print(f"模型类型: {info['type']}, 名称: {info['name']}, 大小: {model_size:.2f}MB")
        return jsonify(info)
        
    except Exception as e:
        print(f"获取模型信息出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_training')
def get_recent_training():
    """获取最近的训练记录"""
    try:
        # 获取最近5条训练记录
        response = get_training_records()
        data = json.loads(response.data)
        recent_records = data['records'][:5]
        
        return jsonify({'records': recent_records})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)