from ultralytics import YOLO
import cv2

# โหลดโมเดลที่ฝึกเอง
print("[STATUS] กำลังโหลดโมเดล...")
model = YOLO(r"runs/detect/train-4/weights/best.pt")
print("[STATUS] โหลดโมเดลเสร็จแล้ว!")

# เปิดกล้อง
print("[STATUS] กำลังเปิดกล้อง...")
cap = cv2.VideoCapture(0)
print("[STATUS] กล้องพร้อมใช้งาน! หน้าต่างจะเปิดขึ้นมาแล้วครับ")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ตรวจจับ
    results = model(frame)

    # วาดกรอบ
    annotated = results[0].plot()

    cv2.imshow("Box Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()