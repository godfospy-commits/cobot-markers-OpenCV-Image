# 01calibrate_polyline.py
#
#

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_HOMOGRAPHY_PATH = os.path.join(BASE_DIR, "calib_homography.npy")
CALIB_ANGLE_PATH = os.path.join(BASE_DIR, "calib_angle_offset.npy")
USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam
MIN_BOX_AREA = 3000

pixel_points = []
world_points = []
angle_offset = None


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


def detect_box_opencv(frame, min_area=MIN_BOX_AREA):
    """
    ตรวจจับกล่องด้วย Pure OpenCV โดยใช้ Color Segmentation (HSV):
    - แยกสีน้ำตาล/เหลืองของกล่องกระดาษ (Hue 5-40, Saturation >= 25)
    - ตัดแสงสะท้อนหลอดไฟ (Glare สีขาว) และพื้นโต๊ะสีขาว/เทาออกทั้งหมด
    - คืนค่าจุดกึ่งกลางกล่อง (Center) และมุมเอียงที่ถูกต้อง
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    lower_brown = np.array([5, 25, 40], dtype=np.uint8)
    upper_brown = np.array([40, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_blur, lower_brown, upper_brown)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None, mask

    largest = max(valid_contours, key=cv2.contourArea)

    bx, by, bw, bh = cv2.boundingRect(largest)

    (rect_cx, rect_cy), (rw, rh), angle = cv2.minAreaRect(largest)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    angle = angle % 180.0
    if angle >= 90.0:
        angle -= 180.0

    box_pts = cv2.boxPoints(((rect_cx, rect_cy), (rw, rh), angle))

    result = {
        "cx": float(rect_cx),
        "cy": float(rect_cy),
        "center_cx": float(rect_cx),
        "center_cy": float(rect_cy),
        "angle_deg": float(angle),
        "box_points": box_pts.astype(np.int32),
        "bbox": (bx, by, bx + bw, by + bh),
        "area": cv2.contourArea(largest),
    }
    return result, mask


def main():
    global angle_offset

    cap = open_camera()
    if cap is None:
        return

    WINDOW_NAME = "Calibrate Box Center+Angle - c=Save, a=Calib Angle, q=Quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print("[STATUS] โหมดตรวจจับ: Pure OpenCV Image Processing (ใช้จุดกึ่งกลางกล่อง Center)")
    print("[STATUS] วิธีใช้:")
    print("  ตำแหน่ง:")
    print("    1) วางกล่องจริงตรงจุดที่ต้องการ, ขยับหุ่นยนต์ไปชี้ตรง 'กึ่งกลางกล่อง', อ่าน X,Y จาก RoboDK, กด 'c' บันทึกจุด")
    print("    2) ทำซ้ำอย่างน้อย 4-6 จุด กระจายทั่วพื้นที่ทำงาน")
    print("  มุม (ทำแค่ 1 ครั้ง):")
    print("    3) วางกล่องให้ขอบขนานกับแกนอ้างอิงของหุ่นยนต์ แล้วกด 'a'")
    print("       ระบบจะถามมุมจริงของหุ่นยนต์ (ปกติพิมพ์ 0)")
    print("  4) กด 'q' เมื่อครบ เพื่อคำนวณและบันทึกไฟล์ calibration ทั้งหมด")
    print("=" * 70 + "\n")

    last_det_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        annotated = frame.copy()
        det_result, mask = detect_box_opencv(frame)
        last_det_result = det_result

        if det_result is not None:
            cx = det_result["cx"]
            cy = det_result["cy"]
            angle_pixel = det_result["angle_deg"]
            box_pts = det_result["box_points"]
            bx1, by1, bx2, by2 = det_result["bbox"]

            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 0, 255), thickness=2)

            cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (int(cx), int(cy)), 3, (255, 255, 255), -1)

            cv2.putText(annotated, f"OpenCV Box (area={det_result['area']:.0f})", (bx1, max(25, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(annotated, f"box center px=({cx:.0f},{cy:.0f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"angle_pixel={angle_pixel:.1f}deg",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(annotated, "ไม่พบกล่องในเฟรมนี้", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_line = f"Saved Points: {len(pixel_points)} | "
        status_line += f"Angle: {'OK (' + format(angle_offset, '.1f') + 'deg)' if angle_offset is not None else 'Not Calibrated'}"
        cv2.putText(annotated, status_line, (10, annotated.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(annotated, "c=Save Point | a=Calib Angle | q=Quit",
                    (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if last_det_result is None:
                print("[WARNING] ไม่พบกล่องในเฟรมปัจจุบัน ขยับกล่อง/ปรับแสงแล้วลองใหม่")
                continue
            cx = last_det_result["cx"]
            cy = last_det_result["cy"]
            try:
                x = float(input(f"  จุดที่ {len(pixel_points) + 1} (กึ่งกลางกล่อง pixel {cx:.0f},{cy:.0f}) -> X จริงใน RoboDK (มม.): "))
                y = float(input(f"  จุดที่ {len(pixel_points) + 1} (กึ่งกลางกล่อง pixel {cx:.0f},{cy:.0f}) -> Y จริงใน RoboDK (มม.): "))
            except ValueError:
                print("[WARNING] ค่าที่กรอกไม่ใช่ตัวเลข ข้ามจุดนี้")
                continue
            pixel_points.append((cx, cy))
            world_points.append((x, y))
            print(f"[POINT] บันทึกจุดที่ {len(pixel_points)} สำเร็จ: pixel center=({cx:.0f},{cy:.0f}) world=({x:.1f},{y:.1f})\n")

        elif key == ord('a'):
            if last_det_result is None:
                print("[WARNING] ไม่พบกล่องในเฟรมปัจจุบัน ลองปรับแสง/ตำแหน่งกล่องแล้วกด 'a' ใหม่")
                continue
            angle_pixel = last_det_result["angle_deg"]
            try:
                robot_angle = float(input(
                    f"  ตรวจพบ angle_pixel = {angle_pixel:.1f} deg\n"
                    f"  พิมพ์มุมจริงของหุ่นยนต์ตอนนี้ (deg, ปกติพิมพ์ 0 ถ้าวางกล่องขนานแกนอ้างอิง): "
                ))
            except ValueError:
                print("[WARNING] ค่าที่กรอกไม่ใช่ตัวเลข ยกเลิกการ calibrate มุมรอบนี้")
                continue
            angle_offset = robot_angle - angle_pixel
            print(f"[ANGLE] Calibrate มุมสำเร็จ: angle_pixel={angle_pixel:.1f} "
                  f"robot_angle={robot_angle:.1f} -> ANGLE_OFFSET={angle_offset:.1f} deg\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print(f"[ERROR] บันทึกตำแหน่งได้แค่ {len(pixel_points)} จุด ต้องการอย่างน้อย 4 จุด -> ไม่บันทึก Homography")
    else:
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        H, status = cv2.findHomography(src, dst)
        np.save(CALIB_HOMOGRAPHY_PATH, H)
        print(f"\n[STATUS] คำนวณ Homography จากจุดกึ่งกลางกล่อง {len(pixel_points)} จุด สำเร็จ!")
        print(f"[STATUS] บันทึกทับไฟล์ {CALIB_HOMOGRAPHY_PATH} เรียบร้อย")
        print(H)

    if angle_offset is None:
        print("[WARNING] ยังไม่ได้ calibrate มุม (ไม่ได้กด 'a') -> ไม่บันทึก calib_angle_offset.npy")
        print("[WARNING] ถ้าจะใช้ระบบหมุนมุมกล่องใน main tracking ต้องรันสคริปต์นี้ใหม่แล้ว calibrate มุมด้วย")
    else:
        np.save(CALIB_ANGLE_PATH, np.array([angle_offset], dtype=np.float32))
        print(f"[STATUS] บันทึก ANGLE_OFFSET = {angle_offset:.2f} deg ลงไฟล์ {CALIB_ANGLE_PATH} เรียบร้อย")


if __name__ == "__main__":
    main()
