from ultralytics import YOLO

if __name__ == '__main__':
    # โหลด Checkpoint ล่าสุดจากการเทรนโฟลเดอร์ train-2
    model = YOLO("runs/detect/train-2/weights/last.pt")

    # สั่งให้เทรนต่อจาก Epoch ค้างไว้
    model.train(resume=True)
