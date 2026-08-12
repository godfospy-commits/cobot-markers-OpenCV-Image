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
BOX_OBJECT_NAME = "BOX"  # object กล่องใน RoboDK ที่จะขยับตามกล่องจริง

# ---------------------------------------------------------
# ปรับ Offset เพื่อให้ BOX ใน RoboDK ตรงกับกล่องจริง
# หน่วยเป็น mm
# +X = ขยับไปทาง X เพิ่ม
# -X = ขยับย้อนทาง X
# +Y = ขยับไปทาง Y เพิ่ม
# -Y = ขยับย้อนทาง Y
# ---------------------------------------------------------
BOX_OFFSET_X = 0.0
BOX_OFFSET_Y = 0.0

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

    # --- ดึง object กล่องจริงมาไว้เทียบตำแหน่งตอน debug (ไม่บังคับต้องมี) ---
    box_obj = RDK.Item(BOX_OBJECT_NAME, robolink.ITEM_TYPE_OBJECT)
    if not box_obj.Valid():
        print(f"[WARNING] ไม่พบ object ชื่อ '{BOX_OBJECT_NAME}' ใน station -> จะข้ามการเทียบตำแหน่ง debug")
        box_obj = None

    # ใช้ Parent ของ pick01 เป็น Tracking Frame เดียวกับพิกัดที่ได้จาก calibration
    tracking_frame = target_pick.Parent()
    if tracking_frame.Valid():
        print(f"[STATUS] Tracking Frame = {tracking_frame.Name()}")
    else:
        raise RuntimeError("ไม่สามารถหา Parent Frame ของ pick01 ได้")

    return RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame


def run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place, world_x, world_y):
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

    # --- เช็คก่อนว่าตำแหน่งใหม่ "เอื้อมถึงไหม" (มีคำตอบ Inverse Kinematics จริงหรือเปล่า) ---
    # ถ้า SolveIK คืนค่าว่าง แปลว่าหุ่นยนต์ไปจุดนั้นด้วยมุมจับเดิมไม่ได้จริงๆ (นอกระยะเอื้อม/ติดข้อจำกัดข้อต่อ)
    sol_pick = robot.SolveIK(new_pick_pose)
    sol_prepick = robot.SolveIK(new_prepick_pose)

    if sol_pick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง pick01 ใหม่ (X={world_x:.1f}, Y={world_y:.1f}) 'เอื้อมไปไม่ถึง' "
              f"ด้วยมุมจับเดิม -> ข้ามรอบนี้ ไม่สั่งหุ่นยนต์เคลื่อนที่")
        return
    if sol_prepick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง prepick01 ใหม่ 'เอื้อมไปไม่ถึง' -> ข้ามรอบนี้ ไม่สั่งหุ่นยนต์เคลื่อนที่")
        return

    print(f"[ROBOT] ตำแหน่งเอื้อมถึงได้ -> ขยับ target ไปตำแหน่งใหม่ X={world_x:.1f}, Y={world_y:.1f}")
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

    RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame = connect_robodk()

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

        now = time.time()
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            # --- กรองเฉพาะ class 'box' (กล่องทั้งใบ) เท่านั้น ---
            # ไม่เอา class0 (ลายพิมพ์/เครื่องหมายในกล่อง) มาปนกัน ไม่งั้นถ้า class0
            # ดันมี confidence สูงกว่า จะถูกเลือกมาใช้แทน ทำให้จับผิดจุด ไม่ตรงกลางกล่อง
            box_cid = next((cid for cid, name in model.names.items() if name == 'box'), 1)
            box_indices = np.where(cls_ids == box_cid)[0]

            if len(box_indices) > 0:
                best_idx = box_indices[np.argmax(confs[box_indices])]
                x1, y1, x2, y2 = xyxy[best_idx]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                world_x, world_y = pixel_to_world(H, cx, cy)

                # ==========================================================
                # ขยับ BOX ใน RoboDK ให้ตามตำแหน่งกล่องจริงแบบ Real-time
                # ใช้ world_x, world_y จาก Homography
                # คงค่า Z และ Rotation เดิมของ BOX
                # ==========================================================
                if box_obj is not None:
                    try:
                        box_pose = robomath.Mat(box_obj.Pose())
                        # world_x/world_y อยู่ใน Tracking Frame เดียวกับ pick01
                        # จึงสร้าง Pose ใน Tracking Frame แล้วแปลงเป็น Absolute Pose
                        # ก่อนส่งให้ BOX เพื่อป้องกันปัญหา BOX อยู่คนละ Parent Frame
                        target_pose = robomath.Mat(box_obj.Pose())
                        target_pose[0, 3] = world_x + BOX_OFFSET_X
                        target_pose[1, 3] = world_y + BOX_OFFSET_Y

                        # ใช้ Z/Rotation เดิมของ BOX
                        box_abs_pose = tracking_frame.PoseAbs() * target_pose
                        box_obj.setPoseAbs(box_abs_pose)
                    except Exception as e:
                        print(f"[WARNING] ขยับ BOX ใน RoboDK ไม่สำเร็จ: {e}")

                # --- วาดจุดแดงกึ่งกลางกล่อง + ตัวเลขพิกัดบนภาพ (แสดงทุกเฟรมที่เจอ) ---
                cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                cv2.putText(annotated, f"px=({cx:.0f},{cy:.0f})", (int(cx) + 12, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated, f"world=({world_x:.1f},{world_y:.1f})mm", (int(cx) + 12, int(cy) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- DEBUG: เทียบกับตำแหน่งจริงของ object กล่องใน RoboDK ---
                if box_obj is not None:
                    try:
                        box_abs = box_obj.PoseAbs()
                        box_rel = tracking_frame.PoseAbs().inv() * box_abs
                        box_x = box_rel[0, 3]
                        box_y = box_rel[1, 3]
                        target_box_x = world_x + BOX_OFFSET_X
                        target_box_y = world_y + BOX_OFFSET_Y
                        err_x = target_box_x - box_x
                        err_y = target_box_y - box_y
                        err_dist = (err_x ** 2 + err_y ** 2) ** 0.5

                        debug_text = f"BOX real=({box_x:.1f},{box_y:.1f}) err=({err_x:+.1f},{err_y:+.1f})={err_dist:.1f}mm"
                        color = (0, 255, 0) if err_dist < 15 else (0, 0, 255)
                        cv2.putText(annotated, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        print(f"[DEBUG] คำนวณ=({world_x:.1f},{world_y:.1f})  BOXจริง=({box_x:.1f},{box_y:.1f})  "
                              f"คลาดเคลื่อน=({err_x:+.1f},{err_y:+.1f}) ระยะ={err_dist:.1f} มม.")
                    except Exception as e:
                        print(f"[WARNING] อ่านตำแหน่ง object '{BOX_OBJECT_NAME}' ไม่สำเร็จ: {e}")

                if (now - last_move_time) > MOVE_COOLDOWN_SEC:
                    print(f"[DETECT] พบวัตถุที่ pixel=({cx:.0f},{cy:.0f}) -> world=({world_x:.1f},{world_y:.1f}) มม.")
                    try:
                        run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place, world_x, world_y)
                    except Exception as e:
                        print(f"[ERROR] สั่งหุ่นยนต์ไม่สำเร็จ: {e}")
                    last_move_time = time.time()

        cv2.imshow("YOLO + RoboDK Dynamic Pick & Place", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
