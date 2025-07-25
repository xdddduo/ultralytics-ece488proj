from ultralytics import YOLO
from ultralytics.nn.tasks import CariesClassifier

model = YOLO("caries/models/yolov8s_trained.pt")  # Load the pre-trained model
if __name__ == "__main__":
  # Access the internal nn.Sequential model
  backbone_and_head = model.model.model

  # Add your custom layer at the end
  new_layer = CariesClassifier()  # initialize your layer
  backbone_and_head.append(new_layer)

  # Re-assign (technically not needed if modifying in-place)
  model.model.model = backbone_and_head
  model.predict(source="caries/datasets/teeth.v1i.yolov8/test/images/6_jpg.rf.20d368e385cf6f248df08c5a6729098e.jpg")

