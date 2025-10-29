from ultralytics import YOLO

# model = RTDETR()
# 加载预训练的COCO RT-DETR-l模型
 
pretrainedModel = 'ultralytics/runs/ChineseMedTrain/exp8/weights/best.pt'

model = YOLO(model=pretrainedModel)

# cfgYAML = 'yolo11RepC3K2SimSPPF_TriPSA.yaml'
# model = YOLO(cfgYAML).load(pretrainedModel)

model.train(
            data='ultralytics/cfg/datasets/originalChineseMed50WithNewLables.yaml', 
            epochs=150, 
            batch=32, 
            imgsz=640)
