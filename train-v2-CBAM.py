from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8-seg-cbam.yaml")
model.train(data="dentex-2-clahe-cropped/data.yaml", epochs=5, imgsz=640)

model = YOLO("runs/segment/trainCBAM/weights/best.pt")
results = model.predict(source="dentex-2-clahe-cropped/valid/images", save=True, conf=0.3)
