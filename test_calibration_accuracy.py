# test_calibration_accuracy.py
# สคริปต์แยกต่างหากสำหรับทดสอบความแม่นยำของ homography (calib_homography.npy)
# โดยเทียบ world_x, world_y ที่ระบบคำนวณได้ กับค่าจริงที่วัดได้ (เช่น jog หุ่นยนต์ไปแตะแล้วอ่านพิกัดจากพาเนล RoboDK)
#
# วิธีใช้:
#   1) วางกล่อง/วัตถุที่ตำแหน่งใดก็ได้ในมุมมองกล้อง
#   2) ดูค่า px, world ที่แสดงบนภาพ realtime
#   3) กด 'c' เพื่อ capture ค่า world_x, world_y ปัจจุบัน
#   4) ไป jog หุ่นยนต์ (หรือวัดด้วยวิธีอื่น) แตะกึ่งกลางกล่อง อ่านค่า X, Y จริงจากพาเนล
#   5) พิมพ์ค่า X, Y จริงลงใน terminal ตามที่ระบบถาม
#   6) ทำซ้ำหลายจุด (แนะนำ 5 จุดขึ้นไป กระจายทั่วเฟรม: กลาง + 4 มุม)
#   7) กด 'q' เพื่อจบและดูตารางสรุป error

import os
import csv
import time
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO

# ------------------- ค่าคงที่ (ให้ตรงกับสคริปต์หลัก) -------------------
MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
CONF_THRESHOLD = 0.5
USB_CAMERA_INDEX = 0
HOMOGRAPHY_FILE = "calib_homography.npy"
OUTPUT_CSV = "calibration_accuracy_report.csv"
# ------------------------------------------------------------------------


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


def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])


def ask_float(prompt):
    """รับ input ตัวเลขจาก terminal พร้อมกันพิมพ์ผิด (พิมพ์ 'skip' เพื่อข้ามจุดนี้)"""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("skip", "s", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            print("  -> พิมพ์ไม่ถูกต้อง กรุณาใส่ตัวเลข (หรือพิมพ์ 'skip' เพื่อข้ามจุดนี้)")


def main():
    if not os.path.exists(HOMOGRAPHY_FILE):
        print(f"[ERROR] ไม่พบ {HOMOGRAPHY_FILE} กรุณารัน calibrate.py ก่อน")
        return
    H = np.load(HOMOGRAPHY_FILE)

    print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

    cap = open_camera()
    if cap is None:
        return

    records = []  # แต่ละรายการ: dict(label, px, py, world_x, world_y, real_x, real_y, err_x, err_y, err_dist)
    last_detection = None  # (cx, cy, world_x, world_y) ของเฟรมล่าสุดที่เจอกล่อง

    print("\n[คำแนะนำการใช้งาน]")
    print("  วางกล่อง -> รอให้กรอบตรวจจับล็อกนิ่ง -> กด 'c' เพื่อ capture จุดนี้")
    print("  จากนั้นไป jog หุ่นยนต์แตะกล่อง อ่านพิกัดจากพาเนล แล้วพิมพ์ค่าจริงตามที่ terminal ถาม")
    print("  กด 'q' เมื่อทดสอบครบทุกจุดแล้ว เพื่อดูตารางสรุป\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        boxes = results[0].boxes

        last_detection = None
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            box_cid = next((cid for cid, name in model.names.items() if name == 'box'), 1)
            box_indices = np.where(cls_ids == box_cid)[0]

            if len(box_indices) > 0:
                best_idx = box_indices[np.argmax(confs[box_indices])]
                x1, y1, x2, y2 = xyxy[best_idx]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                world_x, world_y = pixel_to_world(H, cx, cy)
                last_detection = (cx, cy, world_x, world_y)

                cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                cv2.putText(annotated, f"px=({cx:.0f},{cy:.0f})", (int(cx) + 12, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated, f"world=({world_x:.1f},{world_y:.1f})mm", (int(cx) + 12, int(cy) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_text = f"Captured points: {len(records)}  |  [c]=capture  [q]=quit"
        cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Calibration Accuracy Test", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            if last_detection is None:
                print("[WARNING] ยังไม่เจอกล่องในเฟรมนี้ ลองขยับกล่อง/รอให้ตรวจจับก่อนกด 'c'")
                continue

            cx, cy, world_x, world_y = last_detection
            print(f"\n[CAPTURE] px=({cx:.0f},{cy:.0f})  world=({world_x:.1f},{world_y:.1f}) mm")
            label = input("  ตั้งชื่อจุดนี้ (เช่น 'กลาง', 'มุมซ้ายบน') แล้ว Enter: ").strip() or f"point{len(records)+1}"
            real_x = ask_float("  ใส่ค่า X จริงที่วัดได้ (มม., พิมพ์ 'skip' เพื่อข้าม): ")
            real_y = ask_float("  ใส่ค่า Y จริงที่วัดได้ (มม., พิมพ์ 'skip' เพื่อข้าม): ")

            if real_x is None or real_y is None:
                print("  -> ข้ามจุดนี้ (ไม่บันทึก)\n")
                continue

            err_x = world_x - real_x
            err_y = world_y - real_y
            err_dist = (err_x ** 2 + err_y ** 2) ** 0.5

            records.append({
                "label": label,
                "px": round(cx, 1), "py": round(cy, 1),
                "world_x": round(world_x, 2), "world_y": round(world_y, 2),
                "real_x": round(real_x, 2), "real_y": round(real_y, 2),
                "err_x": round(err_x, 2), "err_y": round(err_y, 2),
                "err_dist": round(err_dist, 2),
            })
            print(f"  -> บันทึกแล้ว! err=({err_x:+.2f},{err_y:+.2f})  ระยะคลาดเคลื่อน={err_dist:.2f} มม.\n")

    cap.release()
    cv2.destroyAllWindows()

    print_summary(records)
    if records:
        save_csv(records)


def print_summary(records):
    print("\n" + "=" * 78)
    print("สรุปผลทดสอบความแม่นยำ Calibration")
    print("=" * 78)

    if not records:
        print("ไม่มีจุดที่บันทึกไว้ (ไม่ได้กด 'c' หรือข้ามทุกจุด)")
        return

    header = f"{'จุด':<14}{'px,py':<16}{'world(x,y)':<20}{'real(x,y)':<20}{'err_dist(mm)':<12}"
    print(header)
    print("-" * 78)
    for r in records:
        print(f"{r['label']:<14}"
              f"({r['px']:.0f},{r['py']:.0f})".ljust(16) +
              f"({r['world_x']:.1f},{r['world_y']:.1f})".ljust(20) +
              f"({r['real_x']:.1f},{r['real_y']:.1f})".ljust(20) +
              f"{r['err_dist']:.2f}".ljust(12))

    err_dists = [r["err_dist"] for r in records]
    print("-" * 78)
    print(f"จำนวนจุดทดสอบ : {len(records)}")
    print(f"error เฉลี่ย   : {sum(err_dists)/len(err_dists):.2f} มม.")
    print(f"error สูงสุด   : {max(err_dists):.2f} มม.  (จุด: "
          f"{records[err_dists.index(max(err_dists))]['label']})")
    print(f"error ต่ำสุด   : {min(err_dists):.2f} มม.")
    print("=" * 78)
    print("แนวทางอ่านผล:")
    print("  - error ทุกจุด <= 3-5mm (แล้วแต่ tolerance งาน pick)  -> homography ใช้ได้")
    print("  - error กลางภาพน้อยแต่ขอบภาพเยอะ -> จุด calibrate กระจายไม่พอ หรือมี lens distortion")
    print("  - error สูงใกล้เคียงกันทุกจุดเหมือน offset คงที่ -> กล้องอาจขยับจากตอน calibrate")
    print("=" * 78 + "\n")


def save_csv(records):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["timestamp", "label", "px", "py", "world_x", "world_y",
                      "real_x", "real_y", "err_x", "err_y", "err_dist"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in records:
            row = {"timestamp": timestamp, **r}
            writer.writerow(row)
    print(f"[STATUS] บันทึกผลลง {OUTPUT_CSV} เรียบร้อย (เพิ่มต่อท้ายไฟล์เดิมถ้ามี)")


if __name__ == "__main__":
    main()
