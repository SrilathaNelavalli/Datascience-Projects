from ultralytics import YOLO
import numpy

model=YOLO("yolov8n.pt","v8")
detection_output=model.predict(source=r"C:\Users\srinu\A VS CODE\YOLO\img1.jpg",conf=0.25,save=True) # it will save image with in folder in 
                                                                                                            # runs\detect\predect

#print(detection_output)
#detection_output[0].save(filename="output1.jpg")# it will save image with in folder with file name output.jpg
print(detection_output[0].numpy())

