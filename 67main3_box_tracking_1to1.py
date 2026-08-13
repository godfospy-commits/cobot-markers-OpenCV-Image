# main3_box_tracking_1to1.py
# USB Camera + YOLO -> Homography -> RoboDK
#
# เป้าหมาย:
#   1) กล่องจริงในกล้องและ BOX ใน RoboDK ขยับสัมพันธ์กัน 1:1
#   2) ใช้ "กึ่งกลาง Bounding Box" เป็นจุดอ้างอิง
#   3) pick01 และ prepick01 ใช้พิกัดเดียวกับ BOX
#   4) UR3 จับตรงกลางกล่อง
#
# สำคัญ:
#   - ต้องรัน prepare_station_1to1.py กับ RoboDK ก่อน
#   - ต้องสร้าง calib_homography.npy ด้วย calibrate_camera_center_1to1.py
#
# Frame ที่ใช้ร่วมกันทั้งระบบ:
#   Orinal_Box
#
# หมายเหตุ:
#   ชื่อ Orinal_Box ใน station สะกดตามไฟล์ RDK เดิมของผู้ใช้

import os
import time
import cv2
import numpy as np

from ultralytics import YOLO
from robodk import robolink, robomath


MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
CONF_THRESHOLD = 0.50
USB_CAMERA_INDEX = 0

ROBOT_NAME = "UR3"
TARGET_PICK_NAME = "pick01"
TARGET_PREPICK_NAME = "prepick01"
PROGRAM_PICK_NAME = "pick"
PROGRAM_PLACE_NAME = "Place"
BOX_OBJECT_NAME = "BOX"

TRACKING_FRAME_NAME = "Orinal_Box"

CALIB_FILE = "calib_homography.npy"

# 0 = ไม่ชดเชยเอง
BOX_OFFSET_X = -50.0
BOX_OFFSET_Y = 100.0
BOX_OFFSET_Z = 0.0

# ถ้า False จะติดตาม BOX และอัปเดต pick01 แต่ไม่สั่ง pick/place อัตโนมัติ
AUTO_PICK = True

# ต้องเห็นกล่องนิ่งต่อเนื่องประมาณนี้ก่อนสั่งงาน
STABLE_REQUIRED = 5
MAX_CENTER_JUMP_PX = 18.0
MOVE_COOLDOWN_SEC = 5.0


def open_camera():
    print(f"[STATUS] เปิดกล้อง USB index={USB_CAMERA_INDEX}")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] เปิดกล้องไม่ได้")
        return None

    # พยายามตั้งความละเอียดให้คงที่
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[STATUS] เปิดกล้องสำเร็จ")
    return cap


def load_homography():
    if not os.path.exists(CALIB_FILE):
        raise FileNotFoundError(
            f"ไม่พบ {CALIB_FILE} กรุณารัน calibrate_camera_center_1to1.py ก่อน"
        )

    H = np.load(CALIB_FILE)

    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise RuntimeError(f"{CALIB_FILE} ไม่ใช่ Homography 3x3 ที่ถูกต้อง")

    return H.astype(np.float64)


def pixel_to_tracking(H, px, py):
    pt = np.array([float(px), float(py), 1.0], dtype=np.float64)
    world = H @ pt

    if abs(world[2]) < 1e-12:
        raise RuntimeError("Homography ให้ค่า denominator ใกล้ 0")

    world /= world[2]
    return float(world[0]), float(world[1])


def connect_robodk():
    RDK = robolink.Robolink()

    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    target_pick = RDK.Item(TARGET_PICK_NAME, robolink.ITEM_TYPE_TARGET)
    target_prepick = RDK.Item(TARGET_PREPICK_NAME, robolink.ITEM_TYPE_TARGET)
    prog_pick = RDK.Item(PROGRAM_PICK_NAME, robolink.ITEM_TYPE_PROGRAM)
    prog_place = RDK.Item(PROGRAM_PLACE_NAME, robolink.ITEM_TYPE_PROGRAM)

    tracking_frame = RDK.Item(TRACKING_FRAME_NAME, robolink.ITEM_TYPE_FRAME)
    box_obj = RDK.Item(BOX_OBJECT_NAME, robolink.ITEM_TYPE_OBJECT)

    for item, name in [
        (robot, ROBOT_NAME),
        (target_pick, TARGET_PICK_NAME),
        (target_prepick, TARGET_PREPICK_NAME),
        (prog_pick, PROGRAM_PICK_NAME),
        (prog_place, PROGRAM_PLACE_NAME),
        (tracking_frame, TRACKING_FRAME_NAME),
    ]:
        if not item.Valid():
            raise RuntimeError(f"ไม่พบ {name} ใน RoboDK")

    if not box_obj.Valid():
        print("[WARNING] ไม่พบ object ชื่อ BOX")
        box_obj = None

    # ทำให้ BOX อยู่ใน Tracking Frame เดียวกับ pick01
    if box_obj is not None:
        parent = box_obj.Parent()
        if not parent.Valid() or parent.Name() != TRACKING_FRAME_NAME:
            print(f"[STATUS] ย้าย BOX ไปอยู่ใต้ Frame '{TRACKING_FRAME_NAME}' แบบรักษาตำแหน่งเดิม")
            box_obj.setParentStatic(tracking_frame)

    # บังคับให้ targets เป็น Cartesian target
    target_pick.setAsCartesianTarget()
    target_prepick.setAsCartesianTarget()

    print(f"[STATUS] Tracking Frame = {tracking_frame.Name()}")
    print("[STATUS] BOX / pick01 / prepick01 ใช้ Tracking Frame เดียวกัน")

    return (
        RDK, robot, target_pick, target_prepick,
        prog_pick, prog_place, box_obj, tracking_frame
    )


def set_xy_keep_z(item, x, y, z_offset=0.0):
    pose = robomath.Mat(item.Pose())
    pose[0, 3] = float(x)
    pose[1, 3] = float(y)
    pose[2, 3] = float(pose[2, 3] + z_offset)
    item.setPose(pose)
    return pose


def update_tracking_items(
    target_pick,
    target_prepick,
    box_obj,
    world_x,
    world_y,
):
    x = world_x + BOX_OFFSET_X
    y = world_y + BOX_OFFSET_Y

    # BOX: local coordinates ใน Orinal_Box
    box_pose = None
    if box_obj is not None:
        box_pose = set_xy_keep_z(box_obj, x, y, BOX_OFFSET_Z)

    # pick01: กลางกล่อง
    pick_pose = set_xy_keep_z(target_pick, x, y, 0.0)

    # prepick01: XY เดียวกัน แต่คง Z เดิม เช่น 200 mm
    prepick_pose = set_xy_keep_z(target_prepick, x, y, 0.0)

    return pick_pose, prepick_pose, box_pose


def run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place):
    pick_pose = target_pick.Pose()
    prepick_pose = target_prepick.Pose()

    sol_pick = robot.SolveIK(pick_pose)
    sol_prepick = robot.SolveIK(prepick_pose)

    if sol_pick.size(1) == 0:
        print("[WARNING] UR3 เอื้อมไม่ถึง pick01 -> ไม่สั่งงาน")
        return False

    if sol_prepick.size(1) == 0:
        print("[WARNING] UR3 เอื้อมไม่ถึง prepick01 -> ไม่สั่งงาน")
        return False

    print("[ROBOT] IK ผ่าน -> รัน pick")
    prog_pick.RunProgram()
    prog_pick.WaitFinished()

    print("[ROBOT] รัน Place")
    prog_place.RunProgram()
    prog_place.WaitFinished()

    print("[ROBOT] pick & place เสร็จ")
    return True


def draw_overlay(frame, x1, y1, x2, y2, cx, cy, world_x, world_y, conf):
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255, 255, 0),
        2
    )

    cv2.circle(frame, (int(cx), int(cy)), 7, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"BOX center=({cx:.0f},{cy:.0f}) conf={conf:.2f}",
        (int(x1), max(20, int(y1) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"RoboDK XY=({world_x:.1f},{world_y:.1f}) mm",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


def main():
    H = load_homography()
    print("[STATUS] โหลด Homography สำเร็จ")

    model = YOLO(MODEL_PATH)
    cap = open_camera()
    if cap is None:
        return

    (
        RDK, robot, target_pick, target_prepick,
        prog_pick, prog_place, box_obj, tracking_frame
    ) = connect_robodk()

    stable_count = 0
    last_center = None
    last_action_time = 0.0

    print("\n========== START 1:1 BOX TRACKING ==========")
    print("จุดอ้างอิง = กึ่งกลาง Bounding Box")
    print(f"Frame = {TRACKING_FRAME_NAME}")
    print(f"AUTO_PICK = {AUTO_PICK}")
    print("กด q เพื่อออก\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            results = model(frame, verbose=False)

            best = None

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for i, conf in enumerate(confs):
                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = xyxy[i]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if best is None or conf > best[0]:
                        best = (float(conf), x1, y1, x2, y2, cx, cy)

            if best is not None:
                conf, x1, y1, x2, y2, cx, cy = best

                try:
                    world_x, world_y = pixel_to_tracking(H, cx, cy)
                except Exception as e:
                    print("[ERROR] แปลงพิกัดไม่ได้:", e)
                    cv2.imshow("YOLO + RoboDK 1:1", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # ตรวจความนิ่งก่อน
                if last_center is not None:
                    jump = float(np.hypot(cx - last_center[0], cy - last_center[1]))
                    if jump <= MAX_CENTER_JUMP_PX:
                        stable_count += 1
                    else:
                        stable_count = 0
                else:
                    stable_count = 1

                last_center = (cx, cy)

                # อัปเดต BOX + pick01 ทุกเฟรม
                update_tracking_items(
                    target_pick,
                    target_prepick,
                    box_obj,
                    world_x,
                    world_y
                )

                print(
                    f"[TRACK] pixel=({cx:.1f},{cy:.1f}) "
                    f"-> RoboDK({world_x:.1f},{world_y:.1f}) "
                    f"stable={stable_count}"
                )

                # สั่ง pick เมื่อกล่องนิ่ง
                now = time.time()
                if (
                    AUTO_PICK
                    and stable_count >= STABLE_REQUIRED
                    and now - last_action_time >= MOVE_COOLDOWN_SEC
                ):
                    print("[ROBOT] กล่องนิ่ง -> เตรียมจับตรงกลาง")
                    if run_pick_and_place(
                        robot,
                        target_pick,
                        target_prepick,
                        prog_pick,
                        prog_place
                    ):
                        last_action_time = time.time()
                    stable_count = 0

                draw_overlay(
                    frame,
                    x1, y1, x2, y2,
                    cx, cy,
                    world_x, world_y,
                    conf
                )

            else:
                stable_count = 0
                last_center = None
                cv2.putText(
                    frame,
                    "BOX NOT FOUND",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("YOLO + RoboDK 1:1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[STATUS] ปิดระบบ")


if __name__ == "__main__":
    main()
