from ultralytics import YOLO
model=YOLO('../runs/detect/train3/weights/best.pt',task='detect',verbose=False)
results = model(source="test_videos/vid1.mp4",show=True,conf=0.4,save=True)
