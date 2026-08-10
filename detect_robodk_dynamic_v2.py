import os
import time
import numpy as np
import cv2
from ultralytics import YOLO

# ตรวจสอบว่ามีไลบรารี robodk หรือไม่
try:
    from robodk import robolink, robomath
    HAS_ROBODK_LIB = True
except ImportError:
    HAS_ROBODK_LIB = False
    print("[INFO] ไม่พบไลบรารี robodk (ระบบจะทำงานในโหมดตรวจจับภาพปกติ)")

# ------------------- ค่าคงที่ & ตั้งค่าโมเดล -------------------
MODEL_PATH = r"runs/detect/train-2/weights/best.pt"
USB_CAMERA_INDEX = 0  # Logi C270 HD WebCam (หรือปรับตามกล้องของคุณ)

ROBOT_NAME = "UR3"
TARGET_PICK_NAME = "pick01"
TARGET_PREPICK_NAME = "prepick01"
PROGRAM_PICK_NAME = "pick"
PROGRAM_PLACE_NAME = "Place"

MOVE_COOLDOWN_SEC = 5  # ระยะเวลารอคอย (วินาที) หลังสั่งหุ่นยนต์ทำงานเสร็จ ก่อนตรวจจับชิ้นถัดไป

print(f"[STATUS] กำลังโหลดโมเดลจาก {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("[STATUS] โหลดโมเดลเสร็จเรียบร้อย!")

# ----------------------------------------------------
# ฟังก์ชันคำนวณ Contrast Stretching (Piecewise Linear Transformation)
# ----------------------------------------------------
def build_contrast_stretch_lut(r1, s1, r2, s2):
    """
    สร้าง Look-Up Table (LUT) สำหรับปรับ Contrast & Brightness ด้วยช่วงจุด (r1, s1) และ (r2, s2)
    """
    if r1 >= r2:
        r2 = r1 + 1  # ป้องกันการหารด้วยศูนย์

    lut = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        if r < r1:
            s = (s1 / r1 * r) if r1 > 0 else s1
        elif r <= r2:
            s = s1 + ((s2 - s1) / (r2 - r1)) * (r - r1)
        else:
            s = s2 + ((255 - s2) / (255 - r2)) * (r - r2) if r2 < 255 else s2
        lut[r] = int(np.clip(s, 0, 255))
    return lut

# ----------------------------------------------------
# ฟังก์ชันคำนวณ IoU (Intersection over Union)
# ----------------------------------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# ----------------------------------------------------
# ฟังก์ชันเกี่ยวกับ RoboDK & Homography Transformation
# ----------------------------------------------------
def pixel_to_world(H, px, py):
    """แปลงพิกัดพิกเซล -> พิกัดจริง (X, Y) มม. โดยใช้ homography จาก calibrate.py"""
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])

def connect_robodk():
    if not HAS_ROBODK_LIB:
        return None, None, None, None, None, None

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

    print("[STATUS] เชื่อมต่อ RoboDK และพบรายการ Items ครบทุกตัวเรียบร้อย!")
    return RDK, robot, target_pick, target_prepick, prog_pick, prog_place

def run_pick_and_place(target_pick, target_prepick, prog_pick, prog_place, world_x, world_y):
    """ขยับ target pick01/prepick01 ไปตำแหน่งใหม่ (คง Z และมุมเดิม) แล้วสั่งรันโปรแกรม pick -> Place"""
    orig_pick_pose = target_pick.Pose()
    orig_prepick_pose = target_prepick.Pose()

    orig_pick_x = orig_pick_pose[0, 3]
    orig_pick_y = orig_pick_pose[1, 3]
    delta_x = world_x - orig_pick_x
    delta_y = world_y - orig_pick_y

    new_pick_pose = robomath.Mat(orig_pick_pose)
    new_pick_pose[0, 3] = world_x
    new_pick_pose[1, 3] = world_y

    new_prepick_pose = robomath.Mat(orig_prepick_pose)
    new_prepick_pose[0, 3] = orig_prepick_pose[0, 3] + delta_x
    new_prepick_pose[1, 3] = orig_prepick_pose[1, 3] + delta_y

    print(f"[ROBOT] ขยับ target ไปตำแหน่งใหม่ X={world_x:.1f}, Y={world_y:.1f} มม.")
    target_pick.setPose(new_pick_pose)
    target_prepick.setPose(new_prepick_pose)

    print("[ROBOT] กำลังรันโปรแกรม pick...")
    prog_pick.RunProgram()
    prog_pick.WaitFinished()

    print("[ROBOT] กำลังรันโปรแกรม Place...")
    prog_place.RunProgram()
    prog_place.WaitFinished()

    target_pick.setPose(orig_pick_pose)
    target_prepick.setPose(orig_prepick_pose)
    print("[ROBOT] Pick & Place เสร็จเรียบร้อย!")

def nothing(x):
    pass

# ----------------------------------------------------
# ฟังก์ชันเปิดกล้อง
# ----------------------------------------------------
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

# ----------------------------------------------------
# ฟังก์ชันหลัก (Main Program)
# ----------------------------------------------------
def main():
    # โหลดไฟล์ Homography Calibration
    H = None
    if os.path.exists("calib_homography.npy"):
        try:
            H = np.load("calib_homography.npy")
            print("[STATUS] โหลดไฟล์ calib_homography.npy สำเร็จ!")
        except Exception as e:
            print(f"[WARNING] ไม่สามารถโหลด calib_homography.npy: {e}")
    else:
        print("[INFO] ไม่พบไฟล์ calib_homography.npy (ระบบตรวจจับภาพจะทำงานปกติ โดยไม่ส่งพิกัดให้ RoboDK)")

    # เชื่อมต่อ RoboDK
    use_robodk = False
    target_pick = target_prepick = prog_pick = prog_place = None
    if H is not None and HAS_ROBODK_LIB:
        try:
            RDK, robot, target_pick, target_prepick, prog_pick, prog_place = connect_robodk()
            use_robodk = True
        except Exception as e:
            print(f"[INFO] ข้ามการเชื่อมต่อ RoboDK: {e}")

    # เปิดใช้งานกล้อง
    cap = open_camera()
    if cap is None:
        print(f"[ERROR] เปิดกล้อง Index {USB_CAMERA_INDEX} ไม่สำเร็จ!")
        return

    WINDOW_NAME = "USB Camera - Object Detection & RoboDK"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # สร้าง Trackbar สไลเดอร์ปรับค่า (r1, s1, r2, s2, Conf)
    cv2.createTrackbar("r1", WINDOW_NAME, 97, 255, nothing)
    cv2.createTrackbar("s1", WINDOW_NAME, 2, 255, nothing)
    cv2.createTrackbar("r2", WINDOW_NAME, 61, 255, nothing)
    cv2.createTrackbar("s2", WINDOW_NAME, 85, 255, nothing)
    cv2.createTrackbar("Conf (%)", WINDOW_NAME, 35, 100, nothing)

    print("\n" + "="*60)
    print("[STATUS] เปิดใช้งานระบบตรวจจับและสไลเดอร์ปรับแสงเรียบร้อยแล้ว!")
    print("  - เลื่อน r1, s1, r2, s2 บนหน้าต่างเพื่อปรับ Contrast & Brightness")
    print("  - เลื่อน Conf (%) เพื่อปรับค่าความมั่นใจในการตรวจจับ (0 - 100%)")
    print("  - กด 'q' เพื่อปิดโปรแกรม")
    print("="*60 + "\n")

    last_move_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        # ดึงค่าจาก Trackbars
        r1 = cv2.getTrackbarPos("r1", WINDOW_NAME)
        s1 = cv2.getTrackbarPos("s1", WINDOW_NAME)
        r2 = cv2.getTrackbarPos("r2", WINDOW_NAME)
        s2 = cv2.getTrackbarPos("s2", WINDOW_NAME)
        conf_val = cv2.getTrackbarPos("Conf (%)", WINDOW_NAME)
        conf_thresh = max(0.10, conf_val / 100.0)

        # ประมวลผลปรับแสง Contrast Stretching ผ่าน LUT
        lut = build_contrast_stretch_lut(r1, s1, r2, s2)
        processed_frame = cv2.LUT(frame, lut)

        # ตรวจจับวัตถุด้วย YOLO
        results = model(processed_frame, conf=conf_thresh, iou=0.45, verbose=False)

        # ----------------------------------------------------
        # กรองจำกัดจำนวนกรอบและเงื่อนไขตามประเภทคลาส
        # ----------------------------------------------------
        boxes = results[0].boxes
        best_world_pos = None

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            selected_indices = []

            # หา Class ID จากชื่อคลาสในโมเดล
            box_cid = next((cid for cid, name in model.names.items() if name == 'box'), 1)
            class0_cid = next((cid for cid, name in model.names.items() if name == 'class0'), 0)

            # 1. เลือกคลาส 'box' เพียง 1 กรอบที่มีค่า conf สูงสุด
            best_box_coords = None
            box_indices = np.where(cls_ids == box_cid)[0]
            if len(box_indices) > 0:
                sorted_box_idx = box_indices[np.argsort(-confs[box_indices])]
                best_box_idx = sorted_box_idx[0]
                selected_indices.append(best_box_idx)
                best_box_coords = xyxy[best_box_idx]  # [bx1, by1, bx2, by2]

            # 2. คลาส 'class0' จะต้องอยู่ใน 'box' และไม่ทับซ้อนกันเอง (IoU > 0.3)
            if best_box_coords is not None:
                bx1, by1, bx2, by2 = best_box_coords
                margin = 5

                # คำนวณพิกัดจริง (World Coordinates X, Y) จากศูนย์กลางของ box
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if H is not None:
                    wx, wy = pixel_to_world(H, bcx, bcy)
                    best_world_pos = (wx, wy)

                class0_indices = np.where(cls_ids == class0_cid)[0]
                valid_class0_indices = []

                for c_idx in class0_indices:
                    cx1, cy1, cx2, cy2 = xyxy[c_idx]
                    center_x = (cx1 + cx2) / 2.0
                    center_y = (cy1 + cy2) / 2.0

                    if (bx1 - margin) <= center_x <= (bx2 + margin) and (by1 - margin) <= center_y <= (by2 + margin):
                        valid_class0_indices.append(c_idx)

                if len(valid_class0_indices) > 0:
                    valid_class0_indices = np.array(valid_class0_indices)
                    sorted_class0_idx = valid_class0_indices[np.argsort(-confs[valid_class0_indices])]

                    final_class0_indices = []
                    for idx in sorted_class0_idx:
                        curr_box = xyxy[idx]
                        is_duplicate = False
                        for prev_idx in final_class0_indices:
                            prev_box = xyxy[prev_idx]
                            if compute_iou(curr_box, prev_box) > 0.3:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            final_class0_indices.append(idx)
                        if len(final_class0_indices) >= 3:
                            break

                    selected_indices.extend(final_class0_indices)

            # แสดงผลเฉพาะกรอบที่ผ่านการคัดเลือก
            if len(selected_indices) > 0:
                filtered_results = results[0][selected_indices]
                annotated = filtered_results.plot()
            else:
                annotated = results[0].plot()
        else:
            annotated = results[0].plot()

        # แสดงพิกัดจริง World Coordinates บนหน้าจอ (ถ้ามี Homography)
        if best_world_pos is not None:
            wx, wy = best_world_pos
            text = f"World: X={wx:.1f}, Y={wy:.1f} mm"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # สั่งหุ่นยนต์ RoboDK Pick & Place ถ้าระยะเวลา cooldown ผ่านไปแล้ว
            now = time.time()
            if use_robodk and (now - last_move_time) > MOVE_COOLDOWN_SEC:
                print(f"[DETECT] พบวัตถุ -> World X={wx:.1f}, Y={wy:.1f} mm")
                try:
                    run_pick_and_place(target_pick, target_prepick, prog_pick, prog_place, wx, wy)
                    last_move_time = time.time()
                except Exception as e:
                    print(f"[ERROR] สั่งงานหุ่นยนต์ไม่สำเร็จ: {e}")

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()