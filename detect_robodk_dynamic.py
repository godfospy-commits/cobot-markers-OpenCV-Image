# detect_robodk_dynamic.py
# ตรวจจับวัตถุด้วย YOLO จากกล้อง USB -> แปลงพิกัดพิกเซลเป็นพิกัดจริงด้วย homography
# -> ขยับ target "pick01" และ "prepick01" ไปตำแหน่งนั้น -> สั่งรันโปรแกรม "pick" แล้วตาม "Place"
#
# ก่อนรัน:
#   1) เปิด RoboDK พร้อม station เดิม (มี robot "UR3", target "pick01"/"prepick01",
#      program "pick"/"Place")
#   2) ต้องรัน calibrate.py มาก่อนแล้ว ได้ไฟล์ calib_homography.npy
#   3) สำคัญ: ตอน calibrate คุณอ่านพิกัด X,Y จากพาเนล RoboDK ต้องอ่านโดยอ้างอิง
#      "เฟรมเดียวกัน" กับที่ target pick01 ใช้อยู่ (ปกติคือเฟรมเดียวกับที่ตั้งค่า
#      Cartesian coordinates ตอน jog หุ่นยนต์) ไม่งั้นตำแหน่งจะเพี้ยน

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO
from robodk import robolink, robomath

# ------------------- ค่าคงที่ที่ต้องเช็ค/แก้ให้ตรงกับ station ของคุณ -------------------
MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
CONF_THRESHOLD = 0.5
USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam

ROBOT_NAME = "UR3"
TARGET_PICK_NAME = "pick01"
TARGET_PREPICK_NAME = "prepick01"
PROGRAM_PICK_NAME = "pick"
PROGRAM_PLACE_NAME = "Place"

MOVE_COOLDOWN_SEC = 5  # หลังหยิบของ 1 ชิ้น รอกี่วินาทีก่อนตรวจจับรอบใหม่
# --------------------------------------------------------------------------------------


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
    """แปลงพิกัดพิกเซล -> พิกัดจริง (X, Y) มม. โดยใช้ homography จาก calibrate.py"""
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])


def connect_robodk():
    RDK = robolink.Robolink()

    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError(f"ไม่พบหุ่นยนต์ชื่อ '{ROBOT_NAME}' ใน station")

    target_pick = RDK.Item(TARGET_PICK_NAME, robolink.ITEM_TYPE_TARGET)
    if not target_pick.Valid():
        raise RuntimeError(f"ไม่พบ target ชื่อ '{TARGET_PICK_NAME}' ใน station")

    target_prepick = RDK.Item(TARGET_PREPICK_NAME, robolink.ITEM_TYPE_TARGET)
    if not target_prepick.Valid():
        raise RuntimeError(f"ไม่พบ target ชื่อ '{TARGET_PREPICK_NAME}' ใน station")

    prog_pick = RDK.Item(PROGRAM_PICK_NAME, robolink.ITEM_TYPE_PROGRAM)
    if not prog_pick.Valid():
        raise RuntimeError(f"ไม่พบโปรแกรมชื่อ '{PROGRAM_PICK_NAME}' ใน station")

    prog_place = RDK.Item(PROGRAM_PLACE_NAME, robolink.ITEM_TYPE_PROGRAM)
    if not prog_place.Valid():
        raise RuntimeError(f"ไม่พบโปรแกรมชื่อ '{PROGRAM_PLACE_NAME}' ใน station")

    print("[STATUS] เชื่อมต่อ RoboDK และพบ items ครบทุกตัวที่ต้องใช้")
    return RDK, robot, target_pick, target_prepick, prog_pick, prog_place


def run_pick_and_place(target_pick, target_prepick, prog_pick, prog_place, world_x, world_y):
    """ขยับ target pick01/prepick01 ไปตำแหน่งใหม่ (คง Z และมุมเดิม) แล้วสั่งรันโปรแกรม pick -> Place"""

    orig_pick_pose = target_pick.Pose()
    orig_prepick_pose = target_prepick.Pose()

    orig_pick_x = orig_pick_pose[0, 3]
    orig_pick_y = orig_pick_pose[1, 3]
    delta_x = world_x - orig_pick_x
    delta_y = world_y - orig_pick_y

    # target pick01: แทนที่ X, Y ใหม่ตรงๆ (คง Z, มุมเดิม)
    new_pick_pose = robomath.Mat(orig_pick_pose)
    new_pick_pose[0, 3] = world_x
    new_pick_pose[1, 3] = world_y

    # target prepick01: ขยับตาม delta เดียวกัน เพื่อให้จุด approach เลื่อนตามจุด pick
    new_prepick_pose = robomath.Mat(orig_prepick_pose)
    new_prepick_pose[0, 3] = orig_prepick_pose[0, 3] + delta_x
    new_prepick_pose[1, 3] = orig_prepick_pose[1, 3] + delta_y

    print(f"[ROBOT] ขยับ target ไปตำแหน่งใหม่ X={world_x:.1f}, Y={world_y:.1f}")
    target_pick.setPose(new_pick_pose)
    target_prepick.setPose(new_prepick_pose)

    print("[ROBOT] กำลังรันโปรแกรม pick...")
    prog_pick.RunProgram()
    prog_pick.WaitFinished()

    print("[ROBOT] กำลังรันโปรแกรม Place...")
    prog_place.RunProgram()
    prog_place.WaitFinished()

    # คืนค่า target กลับตำแหน่งเดิม เผื่อรอบถัดไปคำนวณ delta จากฐานเดิม
    target_pick.setPose(orig_pick_pose)
    target_prepick.setPose(orig_prepick_pose)

    print("[ROBOT] pick & place เสร็จเรียบร้อย")


def main():
    if not os.path.exists("calib_homography.npy"):
        print("[ERROR] ไม่พบ calib_homography.npy กรุณารัน calibrate.py ก่อน")
        return
    H = np.load("calib_homography.npy")

    print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

    RDK, robot, target_pick, target_prepick, prog_pick, prog_place = connect_robodk()

    cap = open_camera()
    if cap is None:
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
        cv2.imshow("YOLO + RoboDK Dynamic Pick & Place", annotated)

        now = time.time()
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0 and (now - last_move_time) > MOVE_COOLDOWN_SEC:
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            world_x, world_y = pixel_to_world(H, cx, cy)
            print(f"[DETECT] พบวัตถุที่ pixel=({cx:.0f},{cy:.0f}) -> world=({world_x:.1f},{world_y:.1f}) มม.")

            try:
                run_pick_and_place(target_pick, target_prepick, prog_pick, prog_place, world_x, world_y)
            except Exception as e:
                print(f"[ERROR] สั่งหุ่นยนต์ไม่สำเร็จ: {e}")

            last_move_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
