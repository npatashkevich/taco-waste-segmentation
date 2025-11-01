# src/export.py
from ultralytics import YOLO

def main():
    model = YOLO("runs/train/taco-seg/weights/best.pt")
    model.export(format="onnx", opset=12)

if __name__ == "__main__":
    main()