# src/infer.py
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/train/taco-seg/weights/best.pt")
    p.add_argument("--source", required=True, help="image/dir/video/rtsp")
    p.add_argument("--save", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=640, save=args.save)
    for r in results:
        print(r.summary())

if __name__ == "__main__":
    main()