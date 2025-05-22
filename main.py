
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route to confirm API is live
@app.get("/")
def root():
    return {"status": "PowerPlay API is live"}

# Load YOLO model
try:
    model = YOLO("models/yolov8s.pt")
    COCO_CLASSES = model.names
except Exception as e:
    model = None
    COCO_CLASSES = []
    print(f"⚠️ YOLO model failed to load: {e}")

# State
app.state.last_center = None
app.state.touch_count = 0

# Utility
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

@app.post("/detect-football/")
async def detect_football(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded"}

        contents = await file.read()
        if not contents:
            return {"error": "Empty file content"}

        image_np = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image format"}

        if model is None:
            return {"error": "Model failed to load on startup"}

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

    except Exception as e:
        return {"error": str(e)}

@app.post("/reset-touches/")
def reset_touches():
    app.state.touch_count = 0
    app.state.last_center = None
    return {"status": "reset"}

@app.get("/touches/")
def get_touches():
    return {"touches": app.state.touch_count}
