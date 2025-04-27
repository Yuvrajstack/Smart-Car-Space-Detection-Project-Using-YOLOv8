import cv2
import pickle
import numpy as np

cap = cv2.VideoCapture('easy1.mp4')

# Variables to hold the points and names
polylines = []
area_names = []
points = []

# Function to capture 4 points from mouse clicks
def click_points(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

        if len(points) == 4:
            name = input("Enter area name: ")
            area_names.append(name)
            polylines.append(points.copy())
            points = []  # Reset for next parking space

# Set mouse callback BEFORE showing the frame
cv2.namedWindow('Select Parking Spot')
cv2.setMouseCallback('Select Parking Spot', click_points)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Draw all saved polylines (ONLY BORDERS)
    for i, poly in enumerate(polylines):
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Only border
        cv2.putText(frame, area_names[i], (poly[0][0], poly[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw points while clicking
    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)

    cv2.imshow('Select Parking Spot', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Save the polylines and area names
with open('freedomtech', 'wb') as f:
    data = {'polylines': polylines, 'area_names': area_names}
    pickle.dump(data, f)

cap.release()
cv2.destroyAllWindows()