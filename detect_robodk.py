# detect_robodk.py
# รวม YOLO object detection (จากกล้อง USB) เข้ากับ RoboDK API
# เพื่อสั่งหุ่นยนต์ไป pick วัตถุที่ตรวจจับได้
#
# ก่อนรัน:
#   1) เปิดโปรแกรม RoboDK ทิ้งไว้ พร้อม station ที่มีหุ่นยนต์โหลดอยู่
#   2) รัน calibrate.py ให้เสร็จก่อน จะได้ไฟล์ calib_homography.npy
#   3) แก้ค่าคงที่ในหัวไฟล์นี้ให้ตรงกับ station ของคุณ (ชื่อหุ่นยนต์, ความสูง Z, ตำแหน่งวางของ)

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO
from robodk import robolink, robomath

# ------------------- ค่าคงที่ที่ต้องแก้ให้ตรงกับ station ของคุณ -------------------
MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
ROBOT_NAME = "UR3"          # ชื่อหุ่นยนต์ใน station RoboDK ของคุณ
CONF_THRESHOLD = 0.5

SAFE_Z = 200.0               # ความสูงปลอดภัยที่หุ่นยนต์เคลื่อนที่ผ่านด้านบนวัตถุ (มม.)
PICK_Z = 30.0                # ความสูงตอนหยิบวัตถุ (มม.) เหนือโต๊ะ
DROP_XYZ = [400, 0, 100]     # ตำแหน่งวางของ (มม.) X, Y, Z อ้างอิงเฟรมเดียวกับ calibration

# มุมเครื่องมือ (tool orientation) ตอนหยิบ -> คว่ำหน้าลง ปรับตามหุ่นยนต์จริง
TOOL_ORIENTATION = robomath.rotx(np.pi) * robomath.rotz(0)

MOVE_COOLDOWN_SEC = 5        # หลังหยิบของ 1 ชิ้น รอกี่วินาทีก่อนตรวจจับรอบใหม่
# --------------------------------------------------------------------------------


USB_CAMERA_INDEX = 1  # Logi C270 HD WebCam (เช็คได้ด้วย list_cameras.py)

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


def pixel_to_world(H, px, py):
    """แปลงพิกัดพิกเซล -> พิกัดจริง (X, Y) มม. โดยใช้ homography จาก calibrate.py"""
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])


def connect_robodk():
    RDK = robolink.Robolink()
    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError(
            f"ไม่พบหุ่นยนต์ชื่อ '{ROBOT_NAME}' ใน RoboDK station "
            f"กรุณาแก้ตัวแปร ROBOT_NAME ให้ตรงกับชื่อจริงใน station"
        )
    print(f"[STATUS] เชื่อมต่อ RoboDK และหุ่นยนต์ '{ROBOT_NAME}' สำเร็จ")
    return RDK, robot


def move_pick_and_place(robot, x, y):
    """สั่งหุ่นยนต์เคลื่อนที่ไปหยิบวัตถุที่ (x, y) แล้วเอาไปวางที่ DROP_XYZ"""
    approach_pick = robomath.transl(x, y, SAFE_Z) * TOOL_ORIENTATION
    at_pick = robomath.transl(x, y, PICK_Z) * TOOL_ORIENTATION
    approach_drop = robomath.transl(DROP_XYZ[0], DROP_XYZ[1], SAFE_Z) * TOOL_ORIENTATION
    at_drop = robomath.transl(DROP_XYZ[0], DROP_XYZ[1], DROP_XYZ[2]) * TOOL_ORIENTATION

    print(f"[ROBOT] กำลังเคลื่อนไปหยิบวัตถุที่ X={x:.1f}, Y={y:.1f}")
    robot.MoveJ(approach_pick)
    robot.MoveL(at_pick)

    # TODO: สั่งเปิด/ปิดกริปเปอร์ตรงนี้ เช่น
    # robot.setDO('gripper', 1)   # หรือ RDK.RunProgram('Gripper_Close') แล้วแต่ setup จริง
    time.sleep(0.5)

    robot.MoveL(approach_pick)
    robot.MoveJ(approach_drop)
    robot.MoveL(at_drop)

    # TODO: สั่งปล่อยวัตถุตรงนี้
    # robot.setDO('gripper', 0)
    time.sleep(0.5)

    robot.MoveL(approach_drop)
    print("[ROBOT] วางวัตถุเสร็จเรียบร้อย")


def main():
    if not os.path.exists("calib_homography.npy"):
        print("[ERROR] ไม่พบ calib_homography.npy กรุณารัน calibrate.py ก่อน")
        return
    H = np.load("calib_homography.npy")

    print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

    RDK, robot = connect_robodk()

    cap = open_camera()
    if cap is None:
        print("[ERROR] เปิดกล้องไม่ได้")
        return

    last_move_time = 0

    print("[STATUS] เริ่มตรวจจับวัตถุ... (กด 'q' เพื่อปิด)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLO + RoboDK Pick & Place", annotated)

        now = time.time()
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0 and (now - last_move_time) > MOVE_COOLDOWN_SEC:
            # เอากล่องที่ confidence สูงสุดมาใช้
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            world_x, world_y = pixel_to_world(H, cx, cy)
            print(f"[DETECT] พบวัตถุที่ pixel=({cx:.0f},{cy:.0f}) -> world=({world_x:.1f},{world_y:.1f}) มม.")

            try:
                move_pick_and_place(robot, world_x, world_y)
            except Exception as e:
                print(f"[ERROR] สั่งหุ่นยนต์ไม่สำเร็จ: {e}")

            last_move_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()