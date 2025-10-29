import sys
import os

# 确保使用项目目录中的ultralytics版本
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ultralytics import YOLO

if __name__ == '__main__':
    # model = RTDETR()
    # 加载预训练的COCO RT-DETR-l模型
     
    pretrainedModel = 'runs/dianziyuanjian_555/train/weights/best.pt'
    
    cfgYAML = 'ultralytics/cfg/models/11/yolo11RepC3K2SimSPPF.yaml'
    # model = YOLO(model=pretrainedModel)
    # 明确指定任务类型为检测和模型规模，避免警告
    model = YOLO(cfgYAML, task='detect').load(pretrainedModel)
    # model = YOLO('yolo11.yaml')
    
    
    # model = YOLO(cfgYAML, pretrained=True, weights=pretrainedModel)
    
    # 训练模型，明确指定任务类型和保存路径
    model.train(
                data='ultralytics/cfg/datasets/dianziyuanjian_555.yaml', 
                epochs=100,
                task='detect',  # 明确指定检测任务
                project='runs/dianziyuanjian_RepC3K2SimSPPF',  # 自定义项目文件夹名称
                name='train',  # 自定义实验名称
                save_period=10  # 每10个epoch保存一次模型
                )
