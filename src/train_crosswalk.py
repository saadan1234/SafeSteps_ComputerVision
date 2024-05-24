from ultralytics import YOLO
model = YOLO('../trained_models/yolov8s.pt')
result=model.train(data="../datasets/data.yaml",batch=16,epochs=5,imgsz=640, device='mps')