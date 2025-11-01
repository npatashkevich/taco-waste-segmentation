# src/eval.py
from ultralytics import YOLO

def main():
    model = YOLO("runs/train/taco-seg/weights/best.pt")
    model.val(data="data/taco.yaml")

if __name__ == "__main__":
    main()