import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

# Load saved parking spaces
with open("freedomtech", "rb") as f:
    data = pickle.load(f)
    polylines, area_names = data['polylines'], data['area_names']

# Load class names (COCO dataset)
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Open video
cap = cv2.VideoCapture('easy1.mp4')

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()

    count += 1
    if count % 3 != 0:
        continue  # Skip frames for faster processing

    # Run YOLO detection
    results = model.predict(frame, stream=True)

    car_centers = []  # list to store car center points

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            class_name = class_list[cls]

            if 'car' in class_name or 'truck' in class_name:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                car_centers.append((cx, cy))

    occupied = []  # list of occupied parking spots
    total_spots = len(polylines)

    # Check occupancy and draw accordingly
    for i, poly in enumerate(polylines):
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        occupied_flag = False

        for (cx, cy) in car_centers:
            result = cv2.pointPolygonTest(pts, (cx, cy), False)
            if result >= 0:
                occupied_flag = True
                break

        # Draw based on occupancy
        if occupied_flag:
            color = (0, 0, 255)  # Red if occupied
            occupied.append(i)
        else:
            color = (0, 255, 0)  # Green if free

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cvzone.putTextRect(frame, f'{area_names[i]}', (poly[0][0], poly[0][1]-10), 1, 1)

    occupied_count = len(occupied)
    free_count = total_spots - occupied_count

    # Count cars detected overall
    total_cars_detected = len(car_centers)

    # Display nice boxes for info
    cvzone.putTextRect(frame, f'Cars Parked: {occupied_count}', (30, 30), scale=2, thickness=3, colorR=(255, 0, 255))
    cvzone.putTextRect(frame, f'Available Slots: {free_count}', (30, 100), scale=2, thickness=3, colorR=(0, 255, 0))

    cv2.imshow('Parking Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()