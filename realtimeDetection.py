import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("http://192.168.29.234:8080/video")
# cap = cv2.VideoCapture(0)

model = YOLO("C:\Custom Object Detection\yolov8_custom\yolov8m_custom.pt")
# results = model("C:/Custom Object Detection/Kamalesh/20210505_112959.jpg")  # predict on an image


while cap.isOpened():
    ret, frame = cap.read()
    
    results= model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()