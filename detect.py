import os
from ultralytics import YOLO
import cv2

# เลือกโหลดโมเดล: สามารถใช้ yolo11m.pt หรือโมเดลที่ฝึกแล้ว เช่น r"runs/detect/train/weights/best.pt"
MODEL_PATH = r"runs/detect/train-2/weights/best.pt" # ใช้โมเดลล่าสุดที่เทรนจากชุดข้อมูล 150 รูป

USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam (ยืนยันจาก test_camera_index.py)

print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

# ฟังก์ชันเปิดกล้อง USB ด้วย DirectShow (ล็อค index ตรงๆ ไม่ไล่ลองกล้องโน้ตบุ๊ก)
def open_camera():
    print(f"[STATUS] กำลังเปิดกล้อง USB Index {USB_CAMERA_INDEX} (Logi C270)...")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[STATUS] เชื่อมต่อกล้อง Index {USB_CAMERA_INDEX} สำเร็จ!")
            return cap, USB_CAMERA_INDEX
        cap.release()
    return None, -1

cap, cam_index = open_camera()

if cap is None:
    print("[ERROR] ไม่สามารถเปิดกล้องได้! กรุณาตรวจสอบว่ามีโปรแกรมอื่นเปิดกล้องค้างไว้หรือไม่ หรือเสียบสาย USB ใหม่")
    exit()

print(f"[STATUS] กล้อง Index {cam_index} พร้อมใช้งานแล้ว! หน้าต่างกำลังจะแสดงขึ้นมา (กด 'q' เพื่อปิด)")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
        break

    # ตรวจจับวัตถุ (กำหนด conf=0.5 เพื่อกรองกรอบที่โมเดลไม่มั่นใจออก)
    results = model(frame, conf=0.5)

    # วาดกรอบบนภาพ
    annotated = results[0].plot()

    cv2.imshow("USB Camera - Object Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
