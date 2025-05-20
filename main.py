from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("models/yolov8s.pt")  # Make sure this path is correct
COCO_CLASSES = model.names

# State tracking
app.state.last_center = None
app.state.touch_count = 0

# Helper function to compute box center
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# POST /detect-football/
@app.post("/detect-football/")
async def detect_football(file: UploadFile = File(...)):
    contents = await file.read()
    image_np = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.4)
    footballs = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = COCO_CLASSES[cls_id]
            if label not in ["sports ball", "suitcase"]:
                continue

            confidence = float(box.conf)
            xyxy = box.xyxy.tolist()[0]
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            if width > 300 or height > 300:
                continue

            footballs.append({
                "label": label,
                "coordinates": xyxy,
                "confidence": confidence
            })

            center = get_center(xyxy)
            if app.state.last_center:
                dist = np.linalg.norm(np.array(center) - np.array(app.state.last_center))
                if dist > 15:
                    app.state.touch_count += 1
            app.state.last_center = center

    return {
        "detections": footballs,
        "touches": app.state.touch_count
    }

# POST /reset-touches/
@app.post("/reset-touches/")
def reset_touches():
    app.state.touch_count = 0
    app.state.last_center = None
    return {"status": "reset"}

# GET /touches/
@app.get("/touches/")
def get_touches():
    return {"touches": app.state.touch_count}
