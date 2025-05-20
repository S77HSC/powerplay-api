# PowerPlay FastAPI YOLOv8 Deployment (Render)

## 📦 What This Is
This is your custom football tracking API using YOLOv8 and FastAPI. It includes:
- /detect-football/ – detects footballs and tracks movement between frames
- /touches/ – returns the number of touches
- /reset-touches/ – resets the count

## 🚀 How to Deploy on Render

1. Go to [https://render.com](https://render.com)
2. Click **New Web Service**
3. Connect your GitHub repo or drag-and-drop this folder after uploading it to a repo
4. Set these values:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Make sure your YOLOv8 model (`yolov8s.pt`) is in the `/models` directory or change the path in `main.py`

## ✅ After Deployment
Your API will be live at:

```
https://your-app-name.onrender.com/detect-football/
https://your-app-name.onrender.com/touches/
```