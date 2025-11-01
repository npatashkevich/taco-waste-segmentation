# app.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("runs/train/taco-seg/weights/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model.predict(img, imgsz=640)
    r = results[0]
    return {
        "boxes": r.boxes.xyxy.tolist(),
        "classes": r.boxes.cls.tolist(),
        "scores": r.boxes.conf.tolist(),
    }