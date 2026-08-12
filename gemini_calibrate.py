# calibrate.py
# สคริปต์คาลิเบรต: หาความสัมพันธ์ระหว่างพิกัดพิกเซลในภาพกล้อง กับพิกัดจริง (X, Y) บนโต๊ะ
# อ้างอิงจาก Reference Frame ของหุ่นยนต์ใน RoboDK (หน่วย มม.)

import cv2
import numpy as np

pixel_points = []
world_points = []

USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam

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

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_points.append((x, y))
        print(f"[POINT] บันทึกพิกเซลจุดที่ {len(pixel_points)}: ({x}, {y})")

def main():
    cap = open_camera()
    if cap is None:
        return

    window_name = "Calibration - คลิกจุดอ้างอิง (กด 'z' เพื่อ Undo / กด 'q' เมื่อเลือกครบแล้ว)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n---ขั้นตอนการ Calibration---")
    print("1. คลิกจุดอ้างอิงอย่างน้อย 4 จุด (แนะนำกระจายให้ทั่ว 4 มุมโต๊ะ)")
    print("2. หากคลิกผิดจุด ให้กดปุ่ม 'z' บนคีย์บอร์ดเพื่อลบจุดล่าสุด")
    print("3. เมื่อเลือกครบแล้ว ให้กดปุ่ม 'q' เพื่อไปขั้นตอนใส่ค่าพิกัดจริง\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพได้")
            break

        # วาดจุดและหมายเลขลำดับลงบนภาพ
        for i, p in enumerate(pixel_points):
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"P{i + 1}", (p[0] + 10, p[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('z'):  # Undo จุดล่าสุด
            if len(pixel_points) > 0:
                removed = pixel_points.pop()
                print(f"[UNDO] ลบจุดที่ {len(pixel_points) + 1}: {removed} เรียบร้อย")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print("[ERROR] ต้องมีอย่างน้อย 4 จุดสำหรับคำนวณ Homography")
        return

    print("\n------------------------------------------------------------")
    print("[STATUS] กรุณาใส่พิกัดจริง (มม.) โดยอ้างอิงจาก Tracking Frame ใน RoboDK")
    print("------------------------------------------------------------")
    
    for i, p in enumerate(pixel_points):
        print(f"\n---> จุดที่ {i + 1} (ตำแหน่งบนภาพ Pixel: x={p[0]}, y={p[1]})")
        x = float(input(f"  ป้อนค่า X จริง (มม.): "))
        y = float(input(f"  ป้อนค่า Y จริง (มม.): "))
        world_points.append((x, y))

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    # คำนวณเมทริกซ์ Homography
    H, status = cv2.findHomography(src, dst)
    np.save("calib_homography.npy", H)
    
    print("\n============================================================")
    print("[SUCCESS] คำนวณ Homography สำเร็จ! บันทึกไฟล์ที่ calib_homography.npy แล้ว")
    print("============================================================")
    print("Homography Matrix:")
    print(H)

if __name__ == "__main__":
    main()