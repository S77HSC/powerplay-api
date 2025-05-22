# yolov8_roboflow_local.py

import cv2
import numpy as np
from ultralytics import YOLO

# Load Roboflow-trained YOLOv8 weights
model = YOLO("roboflow_best.pt")  # Replace with the exact filename if different

# Detection settings
TARGET_CLASS = "football"
CONF_THRESHOLD = 0.4
movement_threshold = 5
last_center = None
touch_count = 0

# Get center of bounding box
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Start webcam
cap = cv2.VideoCapture(0)
print("✅ Webcam started — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    results = model(frame, conf=CONF_THRESHOLD)
    detected = False

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            if label != TARGET_CLASS:
                continue

            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            center = get_center(xyxy)

            # Draw bounding box
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected = True

            if last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(last_center))
                if dist > movement_threshold:
                    touch_count += 1
                    print(f"Touch counted! Distance: {dist:.2f}")
                    last_center = center
            else:
                last_center = center

    if not detected:
        last_center = None

    # Display touch count
    cv2.putText(frame, f"Touches: {touch_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Roboflow YOLOv8 Touch Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
