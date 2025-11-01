# src/train.py
from ultralytics import YOLO

def main():
    # маленькая сегментационная модель
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data="data/taco.yaml",
        imgsz=640,
        epochs=50,
        batch=16,
        device=0,              # GPU:0
        project="runs/train",
        name="taco-seg",
    )

if __name__ == "__main__":
    main()