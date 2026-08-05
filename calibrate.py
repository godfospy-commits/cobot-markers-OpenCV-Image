# calibrate.py
# สคริปต์คาลิเบรต: หาความสัมพันธ์ระหว่างพิกัดพิกเซลในภาพกล้อง กับพิกัดจริง (X, Y) บนโต๊ะ
# ที่หุ่นยนต์ใน RoboDK ใช้งาน (หน่วย มม. อ้างอิงจาก Reference Frame ของหุ่นยนต์)
#
# วิธีใช้:
# 1) วางวัตถุอ้างอิง (เช่น เศษกระดาษ/สกรู) ไว้บนโต๊ะ 4 จุดขึ้นไป ในตำแหน่งที่รู้พิกัดจริงแน่นอน
#    (เช่น ขยับปลายแขนหุ่นยนต์ไปแตะจุดนั้นใน RoboDK แล้วอ่านค่า X,Y จากหน้าจอ)
# 2) รันสคริปต์นี้ กล้องจะเปิดขึ้นมา ให้คลิกที่ตำแหน่งพิกเซลของแต่ละจุดตามลำดับ
# 3) ใส่ค่า X,Y จริง (มม.) ของแต่ละจุดตามที่ระบบถาม
# 4) สคริปต์จะคำนวณ Homography Matrix แล้วบันทึกเป็น calib_homography.npy

import cv2
import numpy as np

pixel_points = []
world_points = []

USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam (ยืนยันจาก test_camera_index.py)

def open_camera():
    print(f"[STATUS] กำลังเปิดกล้อง USB Index {USB_CAMERA_INDEX} (Logi C270)...")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[STATUS] เชื่อมต่อกล้อง Index {USB_CAMERA_INDEX} สำเร็จ!")
            return cap
        cap.release()
    print(f"[ERROR] เปิดกล้อง Index {USB_CAMERA_INDEX} ไม่สำเร็จ")
    return None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_points.append((x, y))
        print(f"[POINT] เก็บพิกเซลจุดที่ {len(pixel_points)}: ({x}, {y})")

def main():
    cap = open_camera()
    if cap is None:
        print("[ERROR] เปิดกล้องไม่ได้")
        return

    cv2.namedWindow("Calibration - คลิกจุดอ้างอิง (กด q เมื่อครบ)")
    cv2.setMouseCallback("Calibration - คลิกจุดอ้างอิง (กด q เมื่อครบ)", mouse_callback)

    print("[STATUS] คลิกที่จุดอ้างอิงในภาพ (แนะนำ 4 จุดขึ้นไป), กด 'q' เมื่อครบ")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, p in enumerate(pixel_points):
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibration - คลิกจุดอ้างอิง (กด q เมื่อครบ)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print("[ERROR] ต้องมีอย่างน้อย 4 จุดสำหรับคำนวณ Homography")
        return

    print("\n[STATUS] กรุณาใส่พิกัดจริง (มม.) ของแต่ละจุด ตามลำดับที่คลิกไว้")
    for i, p in enumerate(pixel_points):
        x = float(input(f"  จุดที่ {i + 1} (pixel {p}) -> X จริง (มม.): "))
        y = float(input(f"  จุดที่ {i + 1} (pixel {p}) -> Y จริง (มม.): "))
        world_points.append((x, y))

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    np.save("calib_homography.npy", H)
    print("\n[STATUS] คำนวณ Homography สำเร็จ! บันทึกไว้ที่ calib_homography.npy")
    print(H)

if __name__ == "__main__":
    main()
