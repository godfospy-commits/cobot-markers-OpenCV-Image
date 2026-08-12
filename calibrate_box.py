# calibrate_box.py
# คาลิเบรตแบบใช้กล่องจริง: วางกล่องจริง (ที่มีความสูง) ตามจุดต่างๆ บนโต๊ะ
# ให้ YOLO ตรวจจับจุดกึ่งกลางกล่องเองจากภาพกล้อง แล้วจับคู่กับพิกัดจริงที่อ่านจาก RoboDK
#
# ข้อดีเทียบกับ calibrate.py เดิม: ไม่ต้องคลิกจุดเอง และ homography ที่ได้จะ "ชดเชย"
# ความเยื้อง (parallax) ที่เกิดจากความสูงของกล่องให้อัตโนมัติ เพราะข้อมูลฝึกมาจาก
# การตรวจจับกล่องจริงโดยตรง ไม่ใช่จุดบนพื้นราบ
#
# วิธีใช้:
# 1) เปิด RoboDK พร้อม station เดิม (ไม่บังคับต้องเปิดก็ได้ ถ้าจะพิมพ์พิกัดเอง)
# 2) รันสคริปต์นี้ กล้องจะเปิดขึ้นมาพร้อมกรอบตรวจจับกล่อง real-time
# 3) วางกล่องจริงตรงจุดที่ 1 (แนะนำเป็นมุม/ขอบพื้นที่ทำงานจริงของหุ่นยนต์)
# 4) ไปที่ RoboDK ขยับหุ่นยนต์ไปแตะจุดกึ่งกลางกล่อง (ที่ระดับพื้นโต๊ะ) อ่านค่า X,Y
# 5) กด 'c' บนหน้าต่างกล้อง เพื่อบันทึกจุดนี้ (ระบบจะถาม X,Y ใน terminal)
# 6) ย้ายกล่องไปจุดถัดไป ทำซ้ำอย่างน้อย 4-6 จุด กระจายให้ทั่วพื้นที่ทำงาน
# 7) กด 'q' เมื่อครบ ระบบจะคำนวณ Homography ใหม่ให้ (บันทึกทับ calib_homography.npy)

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam
CONF_THRESHOLD = 0.5

pixel_points = []
world_points = []


def open_camera():
    print(f"[STATUS] กำลังเปิดกล้อง USB Index {USB_CAMERA_INDEX}...")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[STATUS] เชื่อมต่อกล้อง Index {USB_CAMERA_INDEX} สำเร็จ!")
            return cap
        cap.release()
    print(f"[ERROR] เปิดกล้อง Index {USB_CAMERA_INDEX} ไม่สำเร็จ")
    return None


def detect_box_center(model, frame):
    """ตรวจจับกล่อง (class 'box') ที่ conf สูงสุดในเฟรมปัจจุบัน คืนค่า (cx, cy, conf) หรือ None ถ้าไม่เจอ"""
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, results

    cls_ids = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    box_cid = next((cid for cid, name in model.names.items() if name == 'box'), 1)
    box_indices = np.where(cls_ids == box_cid)[0]
    if len(box_indices) == 0:
        return None, results

    best_idx = box_indices[np.argmax(confs[box_indices])]
    x1, y1, x2, y2 = xyxy[best_idx]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (cx, cy, confs[best_idx]), results


def main():
    print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

    cap = open_camera()
    if cap is None:
        return

    WINDOW_NAME = "Calibrate Box - วางกล่องแล้วกด 'c' เพื่อบันทึกจุด, 'q' เมื่อครบ"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print("[STATUS] วิธีใช้:")
    print("  1) วางกล่องจริงตรงจุดที่ต้องการ (ให้กล้องเห็นกล่องชัดเจน)")
    print("  2) ไปที่ RoboDK อ่านพิกัด X,Y จริงของจุดนั้น (ที่ระดับพื้นโต๊ะ ใต้กล่อง)")
    print("  3) กด 'c' บนหน้าต่างนี้ -> ระบบจะถาม X,Y ใน terminal ให้พิมพ์ค่าที่อ่านได้")
    print("  4) ย้ายกล่องไปจุดถัดไป ทำซ้ำอย่างน้อย 4-6 จุด กระจายให้ทั่วพื้นที่ทำงาน")
    print("  5) กด 'q' เมื่อครบ เพื่อคำนวณ Homography")
    print("=" * 70 + "\n")

    last_detection = None  # (cx, cy)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        detection, results = detect_box_center(model, frame)
        annotated = results[0].plot()

        if detection is not None:
            cx, cy, conf = detection
            last_detection = (cx, cy)
            cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.putText(annotated, f"box center px=({cx:.0f},{cy:.0f}) conf={conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            last_detection = None
            cv2.putText(annotated, "ไม่พบกล่องในเฟรมนี้", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(annotated, f"บันทึกแล้ว {len(pixel_points)} จุด | กด c=บันทึกจุด, q=เสร็จสิ้น",
                    (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if last_detection is None:
                print("[WARNING] ไม่พบกล่องในเฟรมปัจจุบัน ขยับกล่อง/ปรับแสงแล้วลองใหม่")
                continue
            cx, cy = last_detection
            try:
                x = float(input(f"  จุดที่ {len(pixel_points) + 1} (pixel {cx:.0f},{cy:.0f}) -> X จริง (มม.): "))
                y = float(input(f"  จุดที่ {len(pixel_points) + 1} (pixel {cx:.0f},{cy:.0f}) -> Y จริง (มม.): "))
            except ValueError:
                print("[WARNING] ค่าที่กรอกไม่ใช่ตัวเลข ข้ามจุดนี้")
                continue
            pixel_points.append((cx, cy))
            world_points.append((x, y))
            print(f"[POINT] บันทึกจุดที่ {len(pixel_points)} สำเร็จ: pixel=({cx:.0f},{cy:.0f}) world=({x:.1f},{y:.1f})\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print(f"[ERROR] บันทึกได้แค่ {len(pixel_points)} จุด ต้องการอย่างน้อย 4 จุด ลองรันใหม่")
        return

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    np.save("calib_homography.npy", H)
    print(f"\n[STATUS] คำนวณ Homography จากกล่องจริง {len(pixel_points)} จุด สำเร็จ!")
    print("[STATUS] บันทึกทับไฟล์ calib_homography.npy เรียบร้อย (ชดเชยความสูงกล่องแล้ว)")
    print(H)


if __name__ == "__main__":
    main()
