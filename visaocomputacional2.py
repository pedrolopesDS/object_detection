import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

image_path = "cats_and_dogs.jpg"
image = cv2.imread(image_path)

detections = model(image)[0]

for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, confidence, class_id = detection
    label = model.names[int(class_id)]

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"{label} {confidence:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

output_path = "output.jpg"
cv2.imwrite(output_path, image)
