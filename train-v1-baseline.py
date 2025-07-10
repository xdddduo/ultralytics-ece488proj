from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")  # or yolov8m-seg.pt, etc.
model.train(data="dentex-2-clahe-cropped/data.yaml", epochs=5, imgsz=640)

model = YOLO("runs/segment/train9/weights/best.pt") 
results = model.predict(source="dentex-2-clahe-cropped/valid/images", save=True, conf=0.3)
