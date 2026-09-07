# 02main.py
# ระบบตรวจจับกล่อง 360 องศา (PCA Moments + Dual Probe + Homography Vector)
# -> แปลงพิกัดพิกเซลเป็นพิกัดจริงด้วย Homography
# -> ขยับ target "pick01" และ "prepick01" ไปตำแหน่งจุดกึ่งกลางกล่อง + หมุนรอบแกน Z ตามทิศทาง 360° จริง
# -> สั่งรันโปรแกรม "pick" แล้วตามด้วย "Place" ใน RoboDK

import os
import time
import numpy as np
import cv2
from robodk import robolink, robomath

# ------------------- ค่าคงที่และการตั้งค่า -------------------
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
BOX_OBJECT_NAME = "BOX"  # Object กล่องใน RoboDK ที่จะขยับตามกล่องจริง

# ปรับ Offset พิกัดตำแหน่งกล่องใน RoboDK (มม.)
BOX_OFFSET_X = 0.0
BOX_OFFSET_Y = 0.0
BOX_OFFSET_Z = 0.0

# ชดเชยองศาโมเดล 3D เริ่มต้นให้ตรงแนวกับกล่องจริง (90.0 องศา เพื่อให้แนวตั้งและแนวนอนตรงกับภาพกล้อง 100%)
MODEL_ANGLE_OFFSET = 90.0
GRIPPER_ANGLE_OFFSET = 0.0

MAX_ANGLE_DEG = 180.0      # รองรับการหมุนรอบทิศ 360 องศา (-180° ถึง +180°)
INVERT_ROBOT_ANGLE = False # ถ้ากล่องหมุนสวนทางกับของจริง ให้เปลี่ยนเป็น True

MOVE_COOLDOWN_SEC = 5      # ระยะเวลารอระหว่างการหยิบแต่ละรอบ (วินาที)

# ตัวแปรสำหรับจำทิศทางหัวกล่องข้ามเฟรม (Hysteresis Memory ป้องกันการสลับทิศกระพริบ)
last_heading_sign = 1.0
heading_switch_counter = 0


def open_camera():
    """เปิดกล้อง USB ด้วย DirectShow"""
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
    """แปลงพิกัดพิกเซล -> พิกัดจริง (X, Y) มม. โดยใช้ Homography Matrix"""
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])


def load_angle_offset():
    """โหลด ANGLE_OFFSET จาก calib_angle_offset.npy"""
    if not os.path.exists(CALIB_ANGLE_PATH):
        print(f"[WARNING] ไม่พบ {CALIB_ANGLE_PATH} -> ใช้ ANGLE_OFFSET = 0.0")
        return 0.0
    try:
        val = np.load(CALIB_ANGLE_PATH)
        offset = float(val[0])
        print(f"[STATUS] โหลด ANGLE_OFFSET = {offset:.2f} deg")
        return offset
    except Exception as e:
        print(f"[WARNING] ข้อผิดพลาดในการโหลด ANGLE_OFFSET: {e} -> ใช้ 0.0")
        return 0.0


def connect_robodk():
    """เชื่อมต่อ RoboDK ที่เปิดอยู่แล้วโดยไม่เปิดหน้าต่างซ้ำซ้อน"""
    print("[STATUS] กำลังเชื่อมต่อ RoboDK...")
    RDK = robolink.Robolink(robodk_path='')

    if not RDK._is_connected():
        if not RDK.Connect():
            raise RuntimeError("ไม่สามารถเชื่อมต่อกับ RoboDK ได้ (กรุณาเปิดโปรแกรม RoboDK ทิ้งไว้ก่อนรัน)")

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

    box_obj = RDK.Item(BOX_OBJECT_NAME, robolink.ITEM_TYPE_OBJECT)
    if not box_obj.Valid():
        box_obj = RDK.Item(BOX_OBJECT_NAME)
    if not box_obj.Valid():
        print(f"[WARNING] ไม่พบ object ชื่อ '{BOX_OBJECT_NAME}' ใน station -> ข้ามการขยับกล่อง 3D")
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

    # หมุนรอบแกน Z ของโต๊ะ/โลก
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

    # คำนวณ Inverse Kinematics แบบ Elbow-Up
    sol_pick = solve_elbow_up_ik(robot, new_pick_abs, orig_pick_joints)
    sol_prepick = solve_elbow_up_ik(robot, new_prepick_abs, orig_prepick_joints)

    if sol_pick is None or sol_pick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง pick01 ใหม่ (X={world_x:.1f}, Y={world_y:.1f}, "
              f"angle={total_angle:.1f}deg) เอื้อมไปไม่ถึง -> ข้ามรอบนี้")
        return False
    if sol_prepick is None or sol_prepick.size(1) == 0:
        print(f"[WARNING] ตำแหน่ง prepick01 ใหม่ เอื้อมไปไม่ถึง -> ข้ามรอบนี้")
        return False

    print(f"[ROBOT] กำลังขยับแขนกลไปที่มุม {total_angle:.1f}deg (X={world_x:.1f}, Y={world_y:.1f})")
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

    # คืนค่า target กลับตำแหน่งเดิม
    target_pick.setPoseAbs(orig_pick_abs)
    target_pick.setJoints(orig_pick_joints)
    target_prepick.setPoseAbs(orig_prepick_abs)
    target_prepick.setJoints(orig_prepick_joints)

    print("[ROBOT] pick & place เสร็จสมบูรณ์")
    return True


def calculate_pca_moments_angle(contour):
    """
    คำนวณแกนหลักและความเอียงของกล่องด้วย Central Image Moments (PCA):
    - มุมต่อเนื่อง 100% ไม่มีการกระโดด 90 องศาแบบ minAreaRect
    """
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']

    theta_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    vx = np.cos(theta_rad)
    vy = np.sin(theta_rad)

    lambda1 = 0.5 * (mu20 + mu02 + np.sqrt(4.0 * (mu11**2) + (mu20 - mu02)**2))
    length = 2.0 * np.sqrt(lambda1) * 1.732

    return cx, cy, vx, vy, theta_rad, length


def detect_box_360_robust(frame, min_area=MIN_BOX_AREA):
    """
    ตรวจจับกล่องและคำนวณทิศทาง 360 องศา (-180° ถึง +180°):
    1. แยกสีกล่องด้วย HSV
    2. หาแกนหลักด้วย Image Moments (PCA) ป้องกันมุมกระโดด 90°
    3. ตรวจสอบ Probe Points 2 จุดเพื่อระบุทิศทางตัว 'B' (สีแดง/ชมพู)
    4. ใช้ Hysteresis ป้องกันการสลับทิศทางแบบกระพริบ
    """
    global last_heading_sign, heading_switch_counter

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # ช่วงสีน้ำตาล/เหลือง/ส้ม ของกล่องกระดาษ
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

    pca_result = calculate_pca_moments_angle(largest)
    if pca_result is None:
        return None, mask

    cx, cy, vx, vy, theta_rad, length = pca_result

    # จุด Probe ทั้ง 2 ฝั่งตามแนวแกนยาวของกล่อง
    probe_dist = max(35.0, length * 0.28)
    p1_x = int(np.clip(cx + probe_dist * vx, 15, frame.shape[1] - 16))
    p1_y = int(np.clip(cy + probe_dist * vy, 15, frame.shape[0] - 16))

    p2_x = int(np.clip(cx - probe_dist * vx, 15, frame.shape[1] - 16))
    p2_y = int(np.clip(cy - probe_dist * vy, 15, frame.shape[0] - 16))

    # ตรวจจับสีแดง/ชมพูของโลโก้ 'B'
    probe_radius = int(max(15, min(35, length * 0.15)))
    red_mask1 = cv2.inRange(hsv, np.array([0, 45, 30]), np.array([15, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([150, 45, 30]), np.array([180, 255, 255]))
    red_logo_mask = red_mask1 | red_mask2

    def get_probe_score(px, py):
        y1, y2 = max(0, py - probe_radius), min(frame.shape[0], py + probe_radius)
        x1, x2 = max(0, px - probe_radius), min(frame.shape[1], px + probe_radius)
        patch = red_logo_mask[y1:y2, x1:x2]
        return float(cv2.countNonZero(patch))

    score_p1 = get_probe_score(p1_x, p1_y)
    score_p2 = get_probe_score(p2_x, p2_y)

    score_diff = score_p1 - score_p2
    new_heading_sign = last_heading_sign

    if abs(score_diff) > 15:
        target_sign = 1.0 if score_diff > 0 else -1.0
        if target_sign != last_heading_sign:
            heading_switch_counter += 1
            if heading_switch_counter >= 3:
                new_heading_sign = target_sign
                heading_switch_counter = 0
        else:
            heading_switch_counter = 0
    else:
        heading_switch_counter = 0

    last_heading_sign = new_heading_sign

    head_vx = vx * last_heading_sign
    head_vy = vy * last_heading_sign

    full_angle_deg = np.degrees(np.arctan2(head_vy, head_vx))
    full_angle_deg = (full_angle_deg + 180.0) % 360.0 - 180.0

    arrow_len = int(probe_dist * 1.3)
    arrow_end_x = int(cx + arrow_len * head_vx)
    arrow_end_y = int(cy + arrow_len * head_vy)

    rect = cv2.minAreaRect(largest)
    box_pts = cv2.boxPoints(rect).astype(np.int32)

    result = {
        "cx": float(cx),
        "cy": float(cy),
        "angle_deg": float(full_angle_deg),
        "major_angle_deg": float(np.degrees(theta_rad)),
        "head_vx": float(head_vx),
        "head_vy": float(head_vy),
        "box_points": box_pts,
        "p1": (p1_x, p1_y),
        "p2": (p2_x, p2_y),
        "score_p1": score_p1,
        "score_p2": score_p2,
        "heading_sign": last_heading_sign,
        "arrow_end": (arrow_end_x, arrow_end_y),
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

    print("=" * 70)
    print("   ระบบตรวจจับและหยิบวางกล่อง 360 องศา (OpenCV + RoboDK UR3)   ")
    print("=" * 70)

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
            orig_box_z = float(init_box_abs[2, 3])
        except Exception as e:
            print(f"[WARNING] ไม่สามารถอ่าน Pose เริ่มต้นของ BOX: {e}")

    cap = open_camera()
    if cap is None:
        return

    last_move_time = 0.0
    model_angle_offset = MODEL_ANGLE_OFFSET
    print("[STATUS] เริ่มการทำงานระบบ... (กด 'q' หรือ ESC เพื่อออกจากโปรแกรม)")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] ไม่สามารถอ่านภาพจากกล้องได้")
            break

        annotated = frame.copy()
        det_result, mask = detect_box_360_robust(frame)
        now = time.time()

        if det_result is not None:
            raw_cx = det_result["cx"]
            raw_cy = det_result["cy"]
            head_vx = det_result["head_vx"]
            head_vy = det_result["head_vy"]
            box_pts = det_result["box_points"]
            p1 = det_result["p1"]
            p2 = det_result["p2"]
            s1 = det_result["score_p1"]
            s2 = det_result["score_p2"]
            h_sign = det_result["heading_sign"]
            arrow_end = det_result["arrow_end"]

            # แปลงพิกัดโลกจริง (มม.)
            world_x, world_y = pixel_to_world(H, raw_cx, raw_cy)

            # แปลงเวกเตอร์ทิศทางหัวกล่องจากพิกเซล -> พิกัดโลกจริงด้วย Homography Matrix โดยตรง
            w_head_x, w_head_y = pixel_to_world(H, raw_cx + 50.0 * head_vx, raw_cy + 50.0 * head_vy)
            dw_x = w_head_x - world_x
            dw_y = w_head_y - world_y

            # คำนวณมุมจริงในระนาบโลกของหุ่นยนต์ (World Coordinate Frame)
            raw_world_angle = np.degrees(np.arctan2(dw_y, dw_x))

            # คำนวณมุมหุ่นยนต์รอบทิศ (-180° ถึง +180°) พร้อมชดเชยทิศทาง
            robot_angle = raw_world_angle + model_angle_offset
            if INVERT_ROBOT_ANGLE:
                robot_angle = -robot_angle

            robot_angle = (robot_angle + 180.0) % 360.0 - 180.0

            # ==========================================================
            # ขยับและหมุน Object BOX ใน RoboDK แบบ 360 องศา
            # ==========================================================
            if box_obj is not None and orig_box_rot is not None:
                try:
                    rot = robomath.rotz(robomath.pi * robot_angle / 180.0)
                    box_abs_pose = rot * orig_box_rot
                    box_abs_pose[0, 3] = world_x + BOX_OFFSET_X
                    box_abs_pose[1, 3] = world_y + BOX_OFFSET_Y
                    box_abs_pose[2, 3] = orig_box_z + BOX_OFFSET_Z
                    box_obj.setPoseAbs(box_abs_pose)
                except Exception as e:
                    pass

            # --- วาดภาพ Graphics บนหน้าต่าง OpenCV ---
            # 1. กรอบกล่องรอบนอก (ม่วง)
            cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 0, 255), thickness=2)

            # 2. จุดกึ่งกลางกล่อง
            cv2.circle(annotated, (int(raw_cx), int(raw_cy)), 7, (0, 0, 255), -1)
            cv2.circle(annotated, (int(raw_cx), int(raw_cy)), 3, (255, 255, 255), -1)

            # 3. จุด Probe ทั้ง 2 ฝั่ง (จุดสีแดง = ฝั่งที่ตรวจพบโลโก้ 'B', จุดสีน้ำเงิน = ฝั่งตรงข้าม)
            c1 = (0, 0, 255) if h_sign > 0 else (255, 100, 0)
            c2 = (0, 0, 255) if h_sign < 0 else (255, 100, 0)
            cv2.circle(annotated, p1, 12, c1, 2)
            cv2.circle(annotated, p2, 12, c2, 2)
            cv2.putText(annotated, f"B:{int(s1)}", (p1[0] - 15, p1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c1, 1)
            cv2.putText(annotated, f"B:{int(s2)}", (p2[0] - 15, p2[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c2, 1)

            # 4. ลูกศรสีเหลืองชี้ทิศทางหัวกล่อง (ชี้ไปทางโลโก้ 'B')
            cv2.arrowedLine(annotated, (int(raw_cx), int(raw_cy)), arrow_end, (0, 255, 255), 3, tipLength=0.25)

            # 5. ข้อมูลสถานะบนหน้าจอ (HUD)
            time_left = max(0.0, MOVE_COOLDOWN_SEC - (now - last_move_time))
            status_pick = "READY TO PICK" if time_left == 0.0 else f"WAIT COOLDOWN ({time_left:.1f}s)"

            cv2.rectangle(annotated, (5, 5), (460, 110), (0, 0, 0), -1)
            cv2.rectangle(annotated, (5, 5), (460, 110), (0, 255, 0), 1)

            cv2.putText(annotated, f"World (X, Y): ({world_x:.1f}, {world_y:.1f}) mm",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"Angle 360: {robot_angle:.1f} deg",
                        (15, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            cv2.putText(annotated, f"Robot Status: {status_pick}",
                        (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f"Box Area: {det_result['area']:.0f} px",
                        (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ==========================================================
            # สั่งรัน Pick & Place เมื่อพ้นช่วง Cooldown
            # ==========================================================
            if (now - last_move_time) > MOVE_COOLDOWN_SEC:
                print(f"\n[DETECT] ตรวจพบกล่องที่พิกัด ({world_x:.1f}, {world_y:.1f}) มม. | มุม {robot_angle:.1f}°")
                try:
                    success = run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place,
                                                world_x, world_y, robot_angle)
                    if success:
                        last_move_time = time.time()
                except Exception as e:
                    print(f"[ERROR] สั่งหุ่นยนต์ผิดพลาด: {e}")
                    last_move_time = time.time()

        else:
            cv2.putText(annotated, "[NO BOX DETECTED]", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("RoboDK 360 Pick & Place System", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('t'):
            model_angle_offset = (model_angle_offset + 90.0) % 360.0
            print(f"[ANGLE] สลับชดเชยมุม 90° = {model_angle_offset:+.1f} deg")

    cap.release()
    cv2.destroyAllWindows()
    print("[STATUS] ปิดโปรแกรมเรียบร้อย")


if __name__ == "__main__":
    main()
