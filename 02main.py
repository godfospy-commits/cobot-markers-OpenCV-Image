# 02main.py
# ตรวจจับวัตถุด้วย Pure OpenCV (Image Processing) -> แปลงพิกัดพิกเซลเป็นพิกัดจริงด้วย Homography
# -> ขยับ target "pick01" และ "prepick01" ไปตำแหน่งจุดกึ่งกลางกล่อง (Center) + หมุนรอบแกน Z ตามมุมเอียงจริง
# -> สั่งรันโปรแกรม "pick" แล้วตามด้วย "Place" ใน RoboDK

import os
import time
import numpy as np
import cv2
from robodk import robolink, robomath

# ------------------- ค่าคงที่ที่ต้องเช็ค/แก้ให้ตรงกับ station ของคุณ -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_HOMOGRAPHY_PATH = os.path.join(BASE_DIR, "calib_homography.npy")
CALIB_ANGLE_PATH = os.path.join(BASE_DIR, "calib_angle_offset.npy")
USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam
MIN_BOX_AREA = 3000   # พื้นที่ต่ำสุดของกล่อง (พิกเซล) เพื่อกรองสัญญาณรบกวนออก

ROBOT_NAME = "UR3"
TARGET_PICK_NAME = "pick01"
TARGET_PREPICK_NAME = "prepick01"
PROGRAM_PICK_NAME = "pick"
PROGRAM_PLACE_NAME = "Place"
BOX_OBJECT_NAME = "BOX"  # object กล่องใน RoboDK ที่จะขยับตามกล่องจริง

# ---------------------------------------------------------
# ปรับ Offset เพื่อให้ BOX ใน RoboDK ตรงกับกล่องจริง (มม.)
# ---------------------------------------------------------
BOX_OFFSET_X = 0.0
BOX_OFFSET_Y = 0.0

# จำกัดมุมหมุนที่ยอมให้หมุน (องศา)
MAX_ANGLE_DEG = 90.0

# ปรับองศาชดเชยหัวกริปเปอร์:
# - ตั้ง 0.0: ตัวพับสีขาวจะขนานตามแนวยาว ชี้ไปที่กระดาษส่วนที่น้อย (ฝั่งพับ)
# - ตั้ง 180.0: ตัวพับสีขาวจะกลับทิศ 180 องศา
# - ตั้ง 90.0 หรือ -90.0: ตัวพับสีขาวจะขวาง 90 องศา
GRIPPER_ANGLE_OFFSET = 0.0

# ถ้าสังเกตว่ากล่องเอียงซ้าย แต่หุ่นยนต์หมุนเอียงขวา (หมุนกลับทิศ) ให้เปลี่ยนเป็น True
INVERT_ROBOT_ANGLE = False

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
    """แปลงพิกัดพิกเซล -> พิกัดจริง (X, Y) มม. โดยใช้ homography จาก calibrate"""
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])


def load_angle_offset():
    """โหลด ANGLE_OFFSET จาก 01calibrate_polyline.py ถ้าไม่มีไฟล์ -> คืน 0 และเตือน"""
    if not os.path.exists(CALIB_ANGLE_PATH):
        print(f"[WARNING] ไม่พบ {CALIB_ANGLE_PATH} -> จะไม่หมุนมุมกล่อง (ใช้ ANGLE_OFFSET=0)")
        print("[WARNING] ถ้าต้องการหมุนมุมจริง ให้รัน 01calibrate_polyline.py แล้วกด 'a' ก่อน")
        return 0.0
    val = np.load(CALIB_ANGLE_PATH)
    offset = float(val[0])
    print(f"[STATUS] โหลด ANGLE_OFFSET = {offset:.2f} deg")
    return offset


def connect_robodk():
    RDK = robolink.Robolink()
    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        # ค้นหาพอร์ต RoboDK อื่นๆ ที่อาจเปิดอยู่ (เช่น 20501, 20502)
        for p in [20501, 20502, 20503, 20500]:
            try:
                candidate_rdk = robolink.Robolink(port=p)
                cand_robot = candidate_rdk.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
                if cand_robot.Valid():
                    RDK = candidate_rdk
                    robot = cand_robot
                    break
            except Exception:
                pass

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

    box_obj = RDK.Item(BOX_OBJECT_NAME, robolink.ITEM_TYPE_OBJECT)
    if not box_obj.Valid():
        print(f"[WARNING] ไม่พบ object ชื่อ '{BOX_OBJECT_NAME}' ใน station -> จะข้ามการเทียบตำแหน่ง debug")
        box_obj = None

    tracking_frame = target_pick.Parent()
    if tracking_frame.Valid():
        print(f"[STATUS] Tracking Frame = {tracking_frame.Name()}")
    else:
        raise RuntimeError("ไม่สามารถหา Parent Frame ของ pick01 ได้")

    return RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame


def solve_elbow_up_ik(robot, target_pose_abs, ref_joints):
    """
    คำนวณ Inverse Kinematics และคัดกรองเฉพาะท่าทางที่ 'ศอกชี้ขึ้นบน (Elbow Up)'
    และ 'หัวดูดชี้ดิ่งลงพื้น (Joint 5 ~ +90°)' เสมอ ป้องกันแขนหุ่นยนต์ม้วนหรือจมทะลุโต๊ะ
    """
    robot_base_inv = robot.Parent().PoseAbs().inv()
    all_sols = robot.SolveIK_All(robot_base_inv * target_pose_abs)
    if all_sols.size(1) == 0:
        return None

    best_sol = None
    min_dist = float('inf')
    for i in range(all_sols.size(1)):
        j = [all_sols[r, i] for r in range(6)]
        # กรองเฉพาะท่าทางที่ถูกต้อง (Elbow Up / Wrist5 ~ +90° / แขนยกเหนือโต๊ะ)
        if not (65 <= j[4] <= 115):
            continue
        if not (j[2] < -30):
            continue
        if not (j[1] < -30):
            continue
        dist = sum((j[r] - ref_joints[r, 0]) ** 2 for r in range(6))
        if dist < min_dist:
            min_dist = dist
            best_sol = j

    if best_sol is None:
        sol = robot.SolveIK(robot_base_inv * target_pose_abs, ref_joints)
        if sol.size(1) > 0:
            return sol
        return None

    return robomath.Mat(best_sol)


def run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place,
                        world_x, world_y, robot_angle_deg):
    """
    ขยับ target pick01/prepick01 ไปตำแหน่งพิกัดจริง (Absolute Base Frame)
    + หมุนรอบแกน Z ตาม robot_angle_deg และอัปเดต Joint Angles เพื่อให้ Joint 6 หมุนตาม
    แล้วสั่งรันโปรแกรม pick -> Place
    """
    orig_pick_abs = target_pick.PoseAbs()
    orig_prepick_abs = target_prepick.PoseAbs()
    orig_pick_joints = target_pick.Joints()
    orig_prepick_joints = target_prepick.Joints()

    delta_x = world_x - orig_pick_abs[0, 3]
    delta_y = world_y - orig_pick_abs[1, 3]

    # คำนวณมุมหมุนรวม (รวม GRIPPER_ANGLE_OFFSET เพื่อชดเชยหน้ากริปเปอร์)
    total_angle = robot_angle_deg + GRIPPER_ANGLE_OFFSET

    # หมุนรอบแกน Z ของโต๊ะ/โลก (หมุนทิศทางเดียวกับกล่อง 3D)
    rot = robomath.rotz(robomath.pi * total_angle / 180.0)

    orig_pick_rot = robomath.Mat(orig_pick_abs)
    orig_pick_rot[0, 3] = 0
    orig_pick_rot[1, 3] = 0
    orig_pick_rot[2, 3] = 0

    orig_prepick_rot = robomath.Mat(orig_prepick_abs)
    orig_prepick_rot[0, 3] = 0
    orig_prepick_rot[1, 3] = 0
    orig_prepick_rot[2, 3] = 0

    new_pick_abs = rot * orig_pick_rot
    new_pick_abs[0, 3] = world_x
    new_pick_abs[1, 3] = world_y
    new_pick_abs[2, 3] = orig_pick_abs[2, 3]

    new_prepick_abs = rot * orig_prepick_rot
    new_prepick_abs[0, 3] = orig_prepick_abs[0, 3] + delta_x
    new_prepick_abs[1, 3] = orig_prepick_abs[1, 3] + delta_y
    new_prepick_abs[2, 3] = orig_prepick_abs[2, 3]

    # คำนวณ Inverse Kinematics แบบ Elbow-Up (ศอกชี้ขึ้นเหนือโต๊ะเสมอ)
    sol_pick = solve_elbow_up_ik(robot, new_pick_abs, orig_pick_joints)
    sol_prepick = solve_elbow_up_ik(robot, new_prepick_abs, orig_prepick_joints)

    if sol_pick is None or sol_pick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง pick01 ใหม่ (X={world_x:.1f}, Y={world_y:.1f}, "
              f"angle={total_angle:.1f}deg) 'เอื้อมไปไม่ถึง' หรือติดท่าทางผิดปกติ -> ข้ามรอบนี้")
        return
    if sol_prepick is None or sol_prepick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง prepick01 ใหม่ 'เอื้อมไปไม่ถึง' -> ข้ามรอบนี้")
        return

    print(f"[ROBOT] ตำแหน่งเอื้อมถึงได้ -> ขยับ target และหมุน Joint 6 ไปที่มุม {total_angle:.1f}deg "
          f"(X={world_x:.1f}, Y={world_y:.1f})")
    target_pick.setPoseAbs(new_pick_abs)
    target_pick.setJoints(sol_pick)

    target_prepick.setPoseAbs(new_prepick_abs)
    target_prepick.setJoints(sol_prepick)

    print("[ROBOT] กำลังรันโปรแกรม pick...")
    prog_pick.RunProgram()
    prog_pick.WaitFinished()

    print("[ROBOT] กำลังรันโปรแกรม Place...")
    prog_place.RunProgram()
    prog_place.WaitFinished()

    # คืนค่า target กลับตำแหน่งและ Joint เดิม
    target_pick.setPoseAbs(orig_pick_abs)
    target_pick.setJoints(orig_pick_joints)
    target_prepick.setPoseAbs(orig_prepick_abs)
    target_prepick.setJoints(orig_prepick_joints)

    print("[ROBOT] pick & place เสร็จเรียบร้อย")


def detect_box_opencv(frame, min_area=MIN_BOX_AREA):
    """
    ตรวจจับกล่องด้วย Pure OpenCV โดยใช้ Color Segmentation (HSV):
    - แยกสีน้ำตาล/เหลืองของกล่องกระดาษ (Hue 5-40, Saturation >= 25)
    - ตัดแสงสะท้อนหลอดไฟ (Glare สีขาว) และพื้นโต๊ะสีขาว/เทาออกทั้งหมด
    - คืนค่าจุดกึ่งกลางกล่อง (Center) และมุมเอียงที่ถูกต้อง
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # ช่วงสีน้ำตาล/เหลือง/ส้ม ของกล่องกระดาษ
    lower_brown = np.array([5, 25, 40], dtype=np.uint8)
    upper_brown = np.array([40, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_blur, lower_brown, upper_brown)

    # เชื่อมต่อพื้นผิวกล่องให้เต็มแผ่น และลบลวดลาย/ตัวหนังสือบนกล่อง
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

    # Bounding Box (แกนปกติ)
    bx, by, bw, bh = cv2.boundingRect(largest)

    # Rotated Rectangle (Polyline เอียงตามกล่อง)
    (rect_cx, rect_cy), (rw, rh), angle = cv2.minAreaRect(largest)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    angle = angle % 180.0
    if angle >= 90.0:
        angle -= 180.0

    box_pts = cv2.boxPoints(((rect_cx, rect_cy), (rw, rh), angle))

    result = {
        # ใช้จุดกึ่งกลางกล่อง (Center) เป็นหลัก เพื่อความแม่นยำไม่ว่าจะหมุนมุมใด
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
    if not os.path.exists(CALIB_HOMOGRAPHY_PATH):
        print(f"[ERROR] ไม่พบ {CALIB_HOMOGRAPHY_PATH} กรุณารัน 01calibrate_polyline.py ก่อน")
        return
    H = np.load(CALIB_HOMOGRAPHY_PATH)
    angle_offset = load_angle_offset()

    print("[STATUS] โหมดตรวจจับ: Pure OpenCV Image Processing (ใช้จุดกึ่งกลางกล่อง Center)")
    RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame = connect_robodk()

    # บันทึก Rotation และความสูง Z เริ่มต้นของ object BOX
    orig_box_rot = None
    orig_box_z = 0.0
    if box_obj is not None:
        try:
            init_box_abs = box_obj.PoseAbs()
            orig_box_rot = robomath.Mat(init_box_abs)
            orig_box_rot[0, 3] = 0
            orig_box_rot[1, 3] = 0
            orig_box_rot[2, 3] = 0
            orig_box_z = init_box_abs[2, 3]
        except Exception as e:
            print(f"[WARNING] ไม่สามารถอ่าน Pose เริ่มต้นของ BOX: {e}")

    cap = open_camera()
    if cap is None:
        return

    last_move_time = 0
    print("[STATUS] เริ่มตรวจจับวัตถุด้วย OpenCV... (กด 'q' เพื่อปิด)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        annotated = frame.copy()
        det_result, mask = detect_box_opencv(frame)

        now = time.time()

        if det_result is not None:
            cx = det_result["cx"]
            cy = det_result["cy"]
            angle_pixel = det_result["angle_deg"]
            box_pts = det_result["box_points"]
            bx1, by1, bx2, by2 = det_result["bbox"]

            world_x, world_y = pixel_to_world(H, cx, cy)

            # คำนวณมุมหุ่นยนต์
            robot_angle = angle_pixel + angle_offset
            if INVERT_ROBOT_ANGLE:
                robot_angle = -robot_angle
            robot_angle = max(-MAX_ANGLE_DEG, min(MAX_ANGLE_DEG, robot_angle))

            # --- วาดกรอบ OpenCV Bounding Box (สีฟ้า) + Polyline เอียง (สีม่วง) ---
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 0, 255), thickness=2)

            # วาดจุดกึ่งกลางกล่อง (Center) สีแดงเด่นชัด
            cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (int(cx), int(cy)), 3, (255, 255, 255), -1)

            # แสดงข้อมูลบนภาพ
            cv2.putText(annotated, f"OpenCV Box (area={det_result['area']:.0f})", (bx1, max(25, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(annotated, f"center px=({cx:.0f},{cy:.0f}) world=({world_x:.1f},{world_y:.1f})mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"angle={robot_angle:.1f}deg (offset={angle_offset:.1f}deg)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ==========================================================
            # ขยับและหมุนแกน Z ของ BOX ใน RoboDK ตามกล่องจริงแบบ Real-time
            # ==========================================================
            if box_obj is not None and orig_box_rot is not None:
                try:
                    rot = robomath.rotz(robomath.pi * robot_angle / 180.0)
                    box_abs_pose = rot * orig_box_rot
                    box_abs_pose[0, 3] = world_x + BOX_OFFSET_X
                    box_abs_pose[1, 3] = world_y + BOX_OFFSET_Y
                    box_abs_pose[2, 3] = orig_box_z
                    box_obj.setPoseAbs(box_abs_pose)
                except Exception as e:
                    print(f"[WARNING] ขยับ/หมุน BOX ใน RoboDK ไม่สำเร็จ: {e}")

            if (now - last_move_time) > MOVE_COOLDOWN_SEC:
                print(f"[DETECT] พบกล่องที่ pixel center=({cx:.0f},{cy:.0f}) -> "
                      f"world=({world_x:.1f},{world_y:.1f}) มม. robot_angle={robot_angle:.1f}deg")
                try:
                    run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place,
                                        world_x, world_y, robot_angle)
                except Exception as e:
                    print(f"[ERROR] สั่งหุ่นยนต์ไม่สำเร็จ: {e}")
                last_move_time = time.time()

        cv2.imshow("OpenCV + RoboDK Dynamic Pick & Place (No YOLO)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
