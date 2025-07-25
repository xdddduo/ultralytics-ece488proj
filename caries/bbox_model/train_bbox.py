from ultralytics import YOLO

if __name__ == "__main__":
    # Load the YOLO model
  model = YOLO("caries/bbox_model/config.yaml")

  model.train(data="caries/datasets/teeth.v1i.yolov8/data.yaml", epochs=1, imgsz=640)
  # Save the trained model
  model.save("caries/models/yolov8s_trained.pt")
