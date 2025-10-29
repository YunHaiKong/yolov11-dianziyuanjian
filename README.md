# YOLOv11 电子元件检测项目

基于 Ultralytics YOLOv11 的电子元件目标检测系统，支持多种电子元件的自动识别和缺陷检测。

## 🚀 项目简介

本项目是一个完整的电子元件检测解决方案，包含：
- **目标检测**: 使用 YOLOv11 模型检测电子元件
- **缺陷识别**: 识别三极管等元件的完好、缺失、弯曲状态
- **Web界面**: 基于 Flask 的图形化操作界面
- **模型优化**: 多种模型架构调优和训练策略

## 📊 支持检测的电子元件

- **聚酯电容** (Polyester Capacitor)
- **热敏电阻** (Thermistor)
- **三极管状态检测**:
  - 三极管完好 (Transistor Intact)
  - 三极管缺失 (Transistor Missing)
  - 三极管弯曲 (Transistor Bent)

## 🛠️ 项目结构

```
yolov11-dianziyuanjian/
├── app.py                    # Flask Web应用主文件
├── convert_orgindata_to_yolo.py  # 数据格式转换脚本
├── train_*.py               # 各种训练脚本
├── 模型架构调优/            # 模型架构优化实验
│   ├── trainRep.py
│   ├── trainRepSim.py
│   └── ...
├── ultralytics/             # YOLOv11 核心代码
├── datesets/                # 数据集目录
│   ├── dianziyuanjian_4/
│   ├── dianziyuanjian_5/
│   └── ...
├── runs/                    # 训练结果和模型保存
├── static/                  # Web静态资源
├── templates/               # Web模板文件
└── docs/                    # 项目文档
```

## ⚡ 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (可选，推荐使用GPU训练)

### 安装依赖

```bash
pip install ultralytics flask torch torchvision opencv-python pillow matplotlib
```

### 数据准备

1. 将原始数据转换为 YOLO 格式：
```bash
python convert_orgindata_to_yolo.py
```

2. 数据集配置文件位于：`ultralytics/cfg/datasets/`

### 模型训练

#### 基础训练
```bash
python train_5.py          # 5类别训练
python train_55.py         # 55类别训练
```

#### 优化训练
```bash
python train_5_optimized.py  # 带优化的训练脚本
```

#### 模型架构调优
```bash
cd 模型架构调优
python trainRep.py         # RepC3K2架构
python trainRepSim.py      # RepC3K2 + SimAM
```

### Web应用启动

```bash
python app.py
```

访问 http://localhost:5000 使用Web界面

## 📈 训练配置

### 数据集配置

项目支持多种数据集配置：
- `dianziyuanjian_4.yaml`: 4类别检测
- `dianziyuanjian_5.yaml`: 5类别检测
- `dianziyuanjian_55.yaml`: 55类别检测

### 训练参数

- **模型**: YOLOv11n, YOLOv11s
- **训练轮数**: 100-150 epochs
- **图片尺寸**: 640x640
- **批次大小**: 8-32
- **设备**: GPU/CPU

## 🌐 Web界面功能

### 主要功能模块

**首页**: 项目介绍和功能导航

<img width="1268" height="643" alt="图片1" src="https://github.com/user-attachments/assets/c07ab5a7-715f-48b2-9816-210effd27388" />

**模型训练**: 在线训练配置和监控

<img width="1268" height="638" alt="图片2" src="https://github.com/user-attachments/assets/d47d44e3-0738-409c-a697-b6096cb3e349" />

**预测分析**: 图像和视频检测

<img width="1268" height="637" alt="图片3" src="https://github.com/user-attachments/assets/c39c458a-2dce-4fa8-aa20-2d29ef1e6af1" />


**模型测试**: 多模型性能对比

<img width="1268" height="651" alt="图片4" src="https://github.com/user-attachments/assets/cd145ff4-08fe-4c00-ac35-c96949ea17e5" />

**结果分析**: 训练结果可视化

<img width="1268" height="654" alt="图片5" src="https://github.com/user-attachments/assets/596b4169-671f-4178-bc87-6f87a5aa81db" />

### API接口

- `/api/start_training`: 启动模型训练
- `/api/predict`: 图像预测
- `/api/upload`: 文件上传
- `/api/get_results`: 获取训练结果

## 🔧 模型架构优化

### 支持的模型变体

- **YOLOv11**: 基础版本
- **RepC3K2**: 优化的卷积模块
- **RT-DETR**: Transformer架构
- **SimAM**: 注意力机制增强

### 优化策略

- 类别权重平衡
- 数据增强
- 学习率调度
- 早停机制

## 📊 性能指标

项目提供完整的训练监控和性能评估：

- 训练损失曲线
- 验证精度
- mAP指标
- 混淆矩阵
- 类别分布分析

## 🎯 使用示例

### Python API

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 进行预测
results = model.predict('test_image.jpg')

# 显示结果
results[0].show()
```

### 命令行预测

```bash
yolo predict model=runs/detect/train/weights/best.pt source=test_image.jpg
```

## 📁 数据集说明

### 数据格式

- **输入格式**: JSON标注文件
- **输出格式**: YOLO格式 (txt文件)
- **图像格式**: JPG/PNG

### 数据预处理

项目包含完整的数据预处理流程：
- 标注格式转换
- 数据集划分
- 数据增强
- 类别平衡

## 🔍 故障排除

### 常见问题

1. **字体显示问题**: 确保系统中安装了中文字体
2. **GPU内存不足**: 减小批次大小或图片尺寸
3. **依赖冲突**: 使用虚拟环境隔离

### 调试工具

- 训练日志分析
- 内存使用监控
- 性能瓶颈检测

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目基于 AGPL-3.0 许可证开源。

## 🙏 致谢

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- 所有贡献者和数据标注人员

---


**注意**: 本项目仍在积极开发中，API和功能可能会有变动。
