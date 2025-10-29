from ultralytics import RTDETR
from PIL import Image
import cv2

modelPath = "rtdetr-l.pt"
# 加载预训练的COCO RT-DETR-l模型
# model = RTDETR(modelPath)

cfgYAML = 'yolo11WithDETRHead.yaml'
# model = YOLO(model=pretrainedModel)
model = RTDETR(cfgYAML).load(modelPath)

model.train(
            data='ultralytics/cfg/datasets/originalChineseMed50.yaml', 
            epochs=120, 
            batch=16, 
            imgsz=640)
