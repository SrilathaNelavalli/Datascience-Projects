import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load class names from file
with open(r"C:\Users\srinu\A VS CODE\YOLO\utils\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for each class
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Open video file
cap = cv2.VideoCapture(r"C:\Users\srinu\A VS CODE\YOLO\5691192-hd_1920_1080_25fps.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLOv8 prediction
    results = model.predict(source=[frame], conf=0.45, save=False)
    boxes = results[0].boxes

    for box in boxes:
        clsID = int(box.cls.numpy()[0])
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        # Draw bounding box
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 2)

        # Put label text
        label = f"{class_list[clsID]} {round(conf * 100, 1)}%"
        cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
