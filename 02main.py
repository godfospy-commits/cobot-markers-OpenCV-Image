import os
import sys
import time
import numpy as np
import cv2
import math
from robodk import robolink, robomath
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_HOMOGRAPHY_PATH = os.path.join(BASE_DIR, 'calib_homography.npy')
CALIB_ANGLE_PATH = os.path.join(BASE_DIR, 'calib_angle_offset.npy')
TEMPLATE_CANDIDATES = [os.path.join(BASE_DIR, 'template_logo1.png'), os.path.join(BASE_DIR, 'template_logo2.png'), os.path.join(BASE_DIR, 'template_logo.png'), os.path.join(BASE_DIR, 'template_logo2.jpg')]
TEMPLATE_LOGO_PATH = TEMPLATE_CANDIDATES[0]
USB_CAMERA_INDEX = 0
MIN_BOX_AREA = 3000

def load_template_logo():
    templates = []
    source_pairs = [('IMG_1.jpg', 'template_logo1.png', (932, 453, 353, 343)), ('IMG_2.jpg', 'template_logo2.png', (896, 387, 349, 337))]
    for src_name, dst_name, default_crop in source_pairs:
        src_path = os.path.join(BASE_DIR, src_name)
        dst_path = os.path.join(BASE_DIR, dst_name)
        if os.path.exists(src_path) and (not os.path.exists(dst_path)):
            try:
                data = np.fromfile(src_path, dtype=np.uint8)
                src_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if src_img is not None:
                    x, y, w, h = default_crop
                    p = 10
                    crop = src_img[max(0, y - p):min(src_img.shape[0], y + h + p), max(0, x - p):min(src_img.shape[1], x + w + p)]
                    is_ok, buf = cv2.imencode('.png', crop)
                    if is_ok:
                        buf.tofile(dst_path)
                        print(f'[STATUS] Extracted template from {src_name} -> {dst_name} successfully')
            except Exception as e:
                print(f'[WARNING] Failed to extract template from {src_name}: {e}')
    candidates = ['template_logo1.png', 'template_logo2.png', 'template_logo.png', 'template_logo2.jpg']
    loaded = set()
    for fname in candidates:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath) and fname not in loaded:
            try:
                data = np.fromfile(fpath, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    target_w = 32
                    scale = target_w / float(gray.shape[1])
                    target_h = int(gray.shape[0] * scale)
                    tmpl_resized = cv2.resize(gray, (target_w, target_h))
                    templates.append(tmpl_resized)
                    loaded.add(fname)
                    print(f"[STATUS] Loaded template from '{fname}' ({target_w}x{target_h})")
                    if len(templates) >= 2:
                        break
            except Exception as e:
                print(f'[WARNING] Failed to load {fpath}: {e}')
    if templates:
        return templates
    print('[WARNING] No logo template found -> Falling back to Symbol & Ink Density Analysis')
    return None
ROBOT_NAME = 'UR3'
TARGET_PICK_NAME = 'pick01'
TARGET_PREPICK_NAME = 'prepick01'
PROGRAM_PICK_NAME = 'pick'
PROGRAM_PLACE_NAME = 'Place'
BOX_OBJECT_NAME = 'BOX'
BOX_OFFSET_X = 0.0
BOX_OFFSET_Y = 0.0
BOX_OFFSET_Z = 0.0
BOX_FLAT_ROT_X = 90.0
BOX_FLAT_ROT_Z = -90.0
MODEL_ANGLE_OFFSET = 0.0
GRIPPER_ANGLE_OFFSET = 180.0
FLAP_HINGE_OFFSET_MM = 50.0
FLAP_LENGTH_MM = 50.0
PICK_OFFSET_ALONG_ARROW_MM = 10.0  # เลื่อนจุดหยิบไปทางทิศลูกศร 10 mm
INVERT_WORLD_X = True
INVERT_WORLD_Y = True
CALIB_CENTER_X = 341.4
CALIB_CENTER_Y = 79.6
MAX_ANGLE_DEG = 180.0
INVERT_ROBOT_ANGLE = False
H_FLIP_Y = True
MOVE_COOLDOWN_SEC = 10
last_heading_sign = 1.0
heading_switch_counter = 0
SUPPORTED_DICTS = [cv2.aruco.DICT_4X4_1000, cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_5X5_100]
ARUCO_DETECTORS = []
for d_enum in SUPPORTED_DICTS:
    d = cv2.aruco.getPredefinedDictionary(d_enum)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.015
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    ARUCO_DETECTORS.append(cv2.aruco.ArucoDetector(d, p))

def detect_aruco_markers_robust(frame):
    detected = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for det in ARUCO_DETECTORS:
        corners, ids, rejected = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in detected:
                    pts = corners[i][0]
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    detected[mid] = {'corners': pts, 'center': (cx, cy)}
    return detected

def open_camera():
    print(f'[STATUS] Opening USB Camera Index {USB_CAMERA_INDEX}...')
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'[STATUS] Connected to Camera Index {USB_CAMERA_INDEX} successfully!')
            return cap
        cap.release()
    print(f'[ERROR] Failed to open Camera Index {USB_CAMERA_INDEX}')
    return None

def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return (float(world[0]), float(world[1]))

def calculate_real_box_metrics(H, box_points_px):
    world_corners = []
    for pt in box_points_px:
        wx, wy = pixel_to_world(H, pt[0], pt[1])
        world_corners.append((wx, wy))
    world_corners = np.array(world_corners, dtype=np.float32)
    d01 = np.linalg.norm(world_corners[1] - world_corners[0])
    d12 = np.linalg.norm(world_corners[2] - world_corners[1])
    d23 = np.linalg.norm(world_corners[3] - world_corners[2])
    d30 = np.linalg.norm(world_corners[0] - world_corners[3])
    edge1 = (d01 + d23) / 2.0
    edge2 = (d12 + d30) / 2.0
    real_width_mm = min(edge1, edge2)
    real_length_mm = max(edge1, edge2)
    real_area_cm2 = real_width_mm * real_length_mm / 100.0
    center_world = np.mean(world_corners, axis=0)
    world_cx = float(center_world[0])
    world_cy = float(center_world[1])
    return {'world_cx': world_cx, 'world_cy': world_cy, 'width_mm': float(real_width_mm), 'length_mm': float(real_length_mm), 'area_cm2': float(real_area_cm2), 'world_corners': world_corners}

def load_angle_offset():
    if not os.path.exists(CALIB_ANGLE_PATH):
        print(f'[WARNING] File not found: {CALIB_ANGLE_PATH} -> Using ANGLE_OFFSET = 0.0')
        return 0.0
    try:
        val = np.load(CALIB_ANGLE_PATH)
        offset = float(val[0])
        print(f'[STATUS] Loaded ANGLE_OFFSET = {offset:.2f} deg')
        return offset
    except Exception as e:
        print(f'[WARNING] Error loading ANGLE_OFFSET: {e} -> Using 0.0')
        return 0.0

def connect_robodk():
    print('[STATUS] Connecting to RoboDK...')
    RDK = robolink.Robolink(robodk_path='')
    if not RDK._is_connected():
        if not RDK.Connect():
            raise RuntimeError('Cannot connect to RoboDK. Please make sure RoboDK is running.')
    robot = RDK.Item(ROBOT_NAME, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError(f"Robot '{ROBOT_NAME}' not found in station")
    target_pick = RDK.Item(TARGET_PICK_NAME, robolink.ITEM_TYPE_TARGET)
    if not target_pick.Valid():
        raise RuntimeError(f"Target '{TARGET_PICK_NAME}' not found in station")
    target_prepick = RDK.Item(TARGET_PREPICK_NAME, robolink.ITEM_TYPE_TARGET)
    if not target_prepick.Valid():
        raise RuntimeError(f"Target '{TARGET_PREPICK_NAME}' not found in station")
    prog_pick = RDK.Item(PROGRAM_PICK_NAME, robolink.ITEM_TYPE_PROGRAM)
    if not prog_pick.Valid():
        raise RuntimeError(f"Program '{PROGRAM_PICK_NAME}' not found in station")
    prog_place = RDK.Item(PROGRAM_PLACE_NAME, robolink.ITEM_TYPE_PROGRAM)
    if not prog_place.Valid():
        raise RuntimeError(f"Program '{PROGRAM_PLACE_NAME}' not found in station")
    print('[STATUS] RoboDK connected and all required items verified.')
    box_obj = RDK.Item(BOX_OBJECT_NAME, robolink.ITEM_TYPE_OBJECT)
    if not box_obj.Valid():
        box_obj = RDK.Item(BOX_OBJECT_NAME)
    if not box_obj.Valid():
        print(f"[WARNING] Object '{BOX_OBJECT_NAME}' not found in station -> Skipping 3D box visualization")
        box_obj = None
    tracking_frame = target_pick.Parent()
    if tracking_frame.Valid():
        print(f'[STATUS] Tracking Frame = {tracking_frame.Name()}')
    else:
        raise RuntimeError(f'Cannot find Parent Frame of {TARGET_PICK_NAME}')
    return (RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame)

def solve_elbow_up_ik(robot, target_pose_abs, ref_joints):
    robot_base_inv = robot.Parent().PoseAbs().inv()
    all_sols = robot.SolveIK_All(robot_base_inv * target_pose_abs)
    if all_sols.size(1) == 0:
        return None
    best_sol = None
    min_dist = float('inf')
    ref_list = [ref_joints[r, 0] if hasattr(ref_joints, 'size') else ref_joints[r] for r in range(6)]
    for i in range(all_sols.size(1)):
        j = [all_sols[r, i] for r in range(6)]
        # 1. Base Joint 1 must stay in the forward quadrant (prevent 180° arm flips)
        diff_j1 = (j[0] - ref_list[0] + 180.0) % 360.0 - 180.0
        if abs(diff_j1) > 45.0:
            continue
        # 2. Elbow up & wrist down configuration
        if not (65.0 <= j[4] <= 115.0 and j[1] < -30.0 and j[2] < -30.0):
            continue
        # 3. Continuous shortest-path unwrap for Joint 6 (prevent 360° twirls)
        diff_j6 = (j[5] - ref_list[5] + 180.0) % 360.0 - 180.0
        j[5] = ref_list[5] + diff_j6
        if not (-355.0 <= j[5] <= 355.0):
            continue
        # Distance metric: prioritize wrist rotation, keep arm body minimal
        dist = sum(((j[r] - ref_list[r]) ** 2 for r in range(6)))
        if dist < min_dist:
            min_dist = dist
            best_sol = j
    if best_sol is None:
        sol = robot.SolveIK(robot_base_inv * target_pose_abs, ref_joints)
        return sol if sol.size(1) > 0 else None
    return robomath.Mat(best_sol)

def run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place, world_x, world_y, robot_angle_deg, gripper_angle_offset=GRIPPER_ANGLE_OFFSET):
    orig_pick_abs = target_pick.PoseAbs()
    orig_prepick_abs = target_prepick.PoseAbs()
    orig_pick_joints = target_pick.Joints()
    orig_prepick_joints = target_prepick.Joints()
    delta_x = world_x - orig_pick_abs[0, 3]
    delta_y = world_y - orig_pick_abs[1, 3]
    total_angle = robot_angle_deg + gripper_angle_offset
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
    sol_prepick = solve_elbow_up_ik(robot, new_prepick_abs, orig_prepick_joints)
    if sol_prepick is None or sol_prepick.size(1) == 0:
        print(f'[WARNING] Target prepick01 unreachable -> Skipping cycle')
        return False
    # Use sol_prepick as exact reference for sol_pick so joints remain identical during vertical descent
    sol_pick = solve_elbow_up_ik(robot, new_pick_abs, sol_prepick)
    if sol_pick is None or sol_pick.size(1) == 0:
        print(f'[WARNING] Target pick01 unreachable (X={world_x:.1f}, Y={world_y:.1f}, angle={total_angle:.1f}deg) -> Skipping cycle')
        return False
    print(f'[ROBOT] Moving robot arm to angle {total_angle:.1f}deg (Flap offset: {gripper_angle_offset:+.1f}deg) at (X={world_x:.1f}, Y={world_y:.1f})')
    target_pick.setPoseAbs(new_pick_abs)
    target_pick.setJoints(sol_pick)
    target_prepick.setPoseAbs(new_prepick_abs)
    target_prepick.setJoints(sol_prepick)
    print("[ROBOT] Running program 'pick'...")
    prog_pick.RunProgram()
    prog_pick.WaitFinished()
    print("[ROBOT] Running program 'Place'...")
    prog_place.RunProgram()
    prog_place.WaitFinished()
    target_pick.setPoseAbs(orig_pick_abs)
    target_pick.setJoints(orig_pick_joints)
    target_prepick.setPoseAbs(orig_prepick_abs)
    target_prepick.setJoints(orig_prepick_joints)
    print('[ROBOT] Pick and place cycle completed.')
    return True

def calculate_approx_poly_angle(contour):
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps_factor in [0.02, 0.015, 0.025, 0.03, 0.035, 0.04, 0.05]:
        cand = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(cand) == 4:
            approx = cand
            break
    if approx is None or len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect).astype(np.float32)
    else:
        pts = approx.reshape(4, 2).astype(np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(angles)]
    v01 = pts[1] - pts[0]
    v12 = pts[2] - pts[1]
    v23 = pts[3] - pts[2]
    v30 = pts[0] - pts[3]
    len_A = (np.linalg.norm(v01) + np.linalg.norm(v23)) / 2.0
    len_B = (np.linalg.norm(v12) + np.linalg.norm(v30)) / 2.0
    if len_A >= len_B:
        vec = (v01 - v23) / 2.0
        length = float(len_A)
    else:
        vec = (v12 - v30) / 2.0
        length = float(len_B)
    norm_v = np.linalg.norm(vec)
    if norm_v > 0:
        vx = float(vec[0] / norm_v)
        vy = float(vec[1] / norm_v)
    else:
        vx, vy = (1.0, 0.0)
    if vy > 0 or (vy == 0 and vx < 0):
        vx, vy = (-vx, -vy)
    theta_rad = float(np.arctan2(vy, vx))
    u = np.array([vx, vy], dtype=np.float32)
    v = np.array([-vy, vx], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    proj_u = [(pt - center) @ u for pt in pts]
    proj_v = [(pt - center) @ v for pt in pts]
    top_indices = np.where(np.array(proj_u) <= 0)[0]
    bot_indices = np.where(np.array(proj_u) > 0)[0]
    if len(top_indices) == 2 and len(bot_indices) == 2:
        tl_idx = top_indices[0] if proj_v[top_indices[0]] <= proj_v[top_indices[1]] else top_indices[1]
        tr_idx = top_indices[1] if proj_v[top_indices[0]] <= proj_v[top_indices[1]] else top_indices[0]
        bl_idx = bot_indices[0] if proj_v[bot_indices[0]] <= proj_v[bot_indices[1]] else bot_indices[1]
        br_idx = bot_indices[1] if proj_v[bot_indices[0]] <= proj_v[bot_indices[1]] else bot_indices[0]
        ordered_pts = np.float32([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]])
    else:
        ordered_pts = pts
    return (cx, cy, vx, vy, theta_rad, length, ordered_pts.astype(np.int32))

def enhance_image_clahe_gaussian(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    enhanced_lab = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def align_box_corners_vertical(box_pts, vx, vy, crop_w=160, crop_h=320):
    pts = box_pts.astype(np.float32)
    u = np.array([vx, vy], dtype=np.float32)
    v = np.array([-vy, vx], dtype=np.float32)
    center = np.mean(pts, axis=0)
    proj_u = [(pt - center) @ u for pt in pts]
    proj_v = [(pt - center) @ v for pt in pts]
    top_indices = np.where(np.array(proj_u) <= 0)[0]
    bot_indices = np.where(np.array(proj_u) > 0)[0]
    if len(top_indices) != 2 or len(bot_indices) != 2:
        src = pts
    else:
        if proj_v[top_indices[0]] <= proj_v[top_indices[1]]:
            tl_idx, tr_idx = (top_indices[0], top_indices[1])
        else:
            tl_idx, tr_idx = (top_indices[1], top_indices[0])
        if proj_v[bot_indices[0]] <= proj_v[bot_indices[1]]:
            bl_idx, br_idx = (bot_indices[0], bot_indices[1])
        else:
            bl_idx, br_idx = (bot_indices[1], bot_indices[0])
        src = np.float32([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]])
    dst = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    return (src, dst)

def detect_box_360_robust(frame, min_area=MIN_BOX_AREA, template_img=None, detected_markers=None):
    global last_heading_sign, heading_switch_counter
    enhanced_frame = enhance_image_clahe_gaussian(frame)
    lab_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
    lab_blur = cv2.GaussianBlur(lab_frame, (5, 5), 0)
    lower_box_lab = np.array([80, 126, 140], dtype=np.uint8)
    upper_box_lab = np.array([245, 155, 195], dtype=np.uint8)
    mask = cv2.inRange(lab_blur, lower_box_lab, upper_box_lab)
    if detected_markers is not None:
        for mid, m_info in detected_markers.items():
            c_pts = m_info['corners'].astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(c_pts)
            pad = 25
            y1, y2 = max(0, by - pad), min(mask.shape[0], by + bh + pad)
            x1, x2 = max(0, bx - pad), min(mask.shape[1], bx + bw + pad)
            mask[y1:y2, x1:x2] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, mask)
    valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= 85000]
    if not valid_contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            return (None, mask)
    largest = max(valid_contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    poly_result = calculate_approx_poly_angle(largest)
    if poly_result is None:
        return (None, mask)
    cx, cy, vx, vy, theta_rad, length, box_pts = poly_result
    crop_w, crop_h = (160, 320)
    src_pts, dst_pts = align_box_corners_vertical(box_pts, vx, vy, crop_w, crop_h)
    M_warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(enhanced_frame, M_warp, (crop_w, crop_h))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    warped_sharp = cv2.addWeighted(warped_gray, 1.5, warped_blur, -0.5, 0)
    symbol_thresh = cv2.adaptiveThreshold(warped_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    margin_x, margin_y = (12, 18)
    symbol_thresh[:margin_y, :] = 0
    symbol_thresh[-margin_y:, :] = 0
    symbol_thresh[:, :margin_x] = 0
    symbol_thresh[:, -margin_x:] = 0
    kernel_sym = np.ones((3, 3), np.uint8)
    symbol_thresh = cv2.morphologyEx(symbol_thresh, cv2.MORPH_OPEN, kernel_sym)
    half_h = crop_h // 2
    top_ink = cv2.countNonZero(symbol_thresh[0:half_h, :])
    bot_ink = cv2.countNonZero(symbol_thresh[half_h:, :])
    match_score = 0.0
    matched_target_sign = last_heading_sign
    best_symbol_box_warped = None
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    red1 = cv2.inRange(warped_hsv, np.array([0, 60, 60], dtype=np.uint8), np.array([14, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(warped_hsv, np.array([160, 60, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    yel = cv2.inRange(warped_hsv, np.array([15, 80, 110], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    lab_red = cv2.inRange(warped_lab[:, :, 1], 138, 255)
    mask_symbol = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(yel, lab_red))
    mask_symbol[:margin_y, :] = 0
    mask_symbol[-margin_y:, :] = 0
    mask_symbol[:, :margin_x] = 0
    mask_symbol[:, -margin_x:] = 0
    kernel_sym_close = np.ones((5, 5), np.uint8)
    mask_symbol = cv2.morphologyEx(mask_symbol, cv2.MORPH_CLOSE, kernel_sym_close)
    mask_symbol = cv2.morphologyEx(mask_symbol, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sym_contours, _ = cv2.findContours(mask_symbol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b_candidates = []
    for c in sym_contours:
        c_area = cv2.contourArea(c)
        if 60 <= c_area <= 4500:
            cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
            aspect = float(cw_box) / max(1, ch_box)
            if 0.5 <= aspect <= 2.0:
                b_candidates.append((cx_box, cy_box, cw_box, ch_box, c_area))
    best_t_score = 0.0
    best_t_box = None
    if template_img is not None:
        tmpl_list = template_img if isinstance(template_img, list) else [template_img]
        for tmpl in tmpl_list:
            th, tw = tmpl.shape[:2]
            for scale in [0.65, 0.75, 0.85, 0.95, 1.05, 1.2]:
                nw, nh = (int(tw * scale), int(th * scale))
                if nw >= crop_w - 20 or nh >= crop_h - 20 or nw < 14 or (nh < 14):
                    continue
                scaled_tmpl = cv2.resize(tmpl, (nw, nh))
                res = cv2.matchTemplate(warped_sharp, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                _, t_max_val, _, t_max_loc = cv2.minMaxLoc(res)
                if t_max_val > best_t_score:
                    best_t_score = float(t_max_val)
                    best_t_box = (t_max_loc[0], t_max_loc[1], nw, nh)
    if b_candidates:
        bx_c, by_c, bw_c, bh_c, _ = max(b_candidates, key=lambda item: item[4])
        pad = 2
        bx_s = max(0, bx_c - pad)
        by_s = max(0, by_c - pad)
        bw_s = min(crop_w - bx_s, bw_c + pad * 2)
        bh_s = min(crop_h - by_s, bh_c + pad * 2)
        best_symbol_box_warped = (bx_s, by_s, bw_s, bh_s)
        match_score = 0.98
        matched_target_sign = 1.0 if by_c + bh_c // 2 > half_h else -1.0
    elif best_t_score >= 0.28 and best_t_box is not None:
        best_symbol_box_warped = best_t_box
        conf = min(0.98, max(0.65, 0.7 + (best_t_score - 0.3) * 1.0))
        match_score = float(conf)
        tx, ty, tw, th = best_t_box
        matched_target_sign = 1.0 if ty + th // 2 > half_h else -1.0
    else:
        total_ink = top_ink + bot_ink
        if total_ink > 50:
            if top_ink != bot_ink:
                matched_target_sign = 1.0 if bot_ink > top_ink else -1.0
                diff_ratio = abs(top_ink - bot_ink) / float(total_ink)
                match_score = float(min(0.65, 0.4 + diff_ratio * 0.4))
            active_roi_y1 = half_h if matched_target_sign == 1.0 else 0
            active_roi_y2 = crop_h if matched_target_sign == 1.0 else half_h
            active_mask = np.zeros_like(symbol_thresh)
            active_mask[active_roi_y1:active_roi_y2, :] = symbol_thresh[active_roi_y1:active_roi_y2, :]
            sym_contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if sym_contours:
                largest_sym = max(sym_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_sym) > 40:
                    sx, sy, sw, sh = cv2.boundingRect(largest_sym)
                    best_symbol_box_warped = (sx, sy, sw, sh)
    if matched_target_sign != last_heading_sign:
        heading_switch_counter += 1
        if heading_switch_counter >= 7:
            last_heading_sign = matched_target_sign
            heading_switch_counter = 0
    else:
        heading_switch_counter = 0
    head_vx = vx * last_heading_sign
    head_vy = vy * last_heading_sign
    full_angle_deg = np.degrees(np.arctan2(head_vy, head_vx))
    full_angle_deg = (full_angle_deg + 180.0) % 360.0 - 180.0
    arrow_len = int(length * 0.4)
    arrow_end_x = int(cx + arrow_len * head_vx)
    arrow_end_y = int(cy + arrow_len * head_vy)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    logo_box_pts = None
    if best_symbol_box_warped is not None:
        lx, ly, lw, lh = best_symbol_box_warped
        l_pts_w = np.float32([[[lx, ly]], [[lx + lw, ly]], [[lx + lw, ly + lh]], [[lx, ly + lh]]])
        l_pts_orig = cv2.perspectiveTransform(l_pts_w, M_inv)
        logo_box_pts = l_pts_orig.reshape(-1, 2).astype(np.int32)
    symbol_mask_orig = cv2.warpPerspective(symbol_thresh, M_inv, (frame.shape[1], frame.shape[0]))
    result = {'cx': float(cx), 'cy': float(cy), 'angle_deg': float(full_angle_deg), 'major_angle_deg': float(np.degrees(theta_rad)), 'head_vx': float(head_vx), 'head_vy': float(head_vy), 'box_points': box_pts, 'heading_sign': last_heading_sign, 'arrow_end': (arrow_end_x, arrow_end_y), 'match_score': match_score, 'logo_box_pts': logo_box_pts, 'best_symbol_box_warped': best_symbol_box_warped, 'symbol_mask_orig': symbol_mask_orig, 'warped_box': warped, 'bbox': (bx, by, bx + bw, by + bh), 'area': cv2.contourArea(largest), 'box_length_px': float(length)}
    return (result, mask)

def main():
    if not os.path.exists(CALIB_HOMOGRAPHY_PATH):
        print(f'[ERROR] File not found: {CALIB_HOMOGRAPHY_PATH}. Please run 01calibrate_polyline.py first.')
        return
    H = np.load(CALIB_HOMOGRAPHY_PATH)
    angle_offset = load_angle_offset()
    template_img = load_template_logo()
    print('=' * 70)
    print('   RoboDK UR3 360-Degree Box Detection & Pick-and-Place System   ')
    print('=' * 70)
    RDK, robot, target_pick, target_prepick, prog_pick, prog_place, box_obj, tracking_frame = connect_robodk()
    orig_box_rot = None
    orig_box_z = 0.0
    if box_obj is not None:
        try:
            init_box_abs = box_obj.PoseAbs()
            flat_rot = robomath.rotz(robomath.pi * BOX_FLAT_ROT_Z / 180.0) * robomath.rotx(robomath.pi * BOX_FLAT_ROT_X / 180.0)
            orig_box_rot = robomath.Mat(flat_rot)
            orig_box_z = float(init_box_abs[2, 3])
            init_flat_pose = robomath.Mat(flat_rot)
            init_flat_pose[0, 3] = float(init_box_abs[0, 3])
            init_flat_pose[1, 3] = float(init_box_abs[1, 3])
            init_flat_pose[2, 3] = orig_box_z + BOX_OFFSET_Z
            box_obj.setPoseAbs(init_flat_pose)
            print(f'[STATUS] BOX forced flat on table (Rx={BOX_FLAT_ROT_X}°, Z={orig_box_z:.1f} mm)')
        except Exception as e:
            print(f'[WARNING] Cannot read initial pose of BOX: {e}')
    cap = open_camera()
    if cap is None:
        return
    last_move_time = 0.0
    model_angle_offset = angle_offset if angle_offset != 0.0 else MODEL_ANGLE_OFFSET
    gripper_angle_offset = GRIPPER_ANGLE_OFFSET
    invert_world_x = INVERT_WORLD_X
    invert_world_y = INVERT_WORLD_Y
    show_hud = True
    show_debug = False
    show_mask = False
    h_flip_y = H_FLIP_Y
    print('[STATUS] System running...')
    print('  [S]=Save Template | [H]=HUD | [D]=Debug Overlay | [M]=Mask View | [Q]=Quit')
    print('  [T]=+90° offset | [ ]=+5° | [ ]=-5° | [F]=Flip Y axis')
    smooth_world_x = None
    smooth_world_y = None
    smooth_sin_ang = None
    smooth_cos_ang = None
    smooth_box_pts = None
    smooth_geom_cx = None
    smooth_geom_cy = None
    smooth_real_w = None
    smooth_real_l = None
    smooth_real_area = None
    smooth_arrow_end = None
    smooth_logo_box_pts = None
    smooth_match_score = None
    smooth_pick_world_x = None
    smooth_pick_world_y = None
    smooth_pick_px = None
    smooth_pick_py = None
    smooth_offset_mm = None
    no_box_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('[WARNING] Failed to capture frame from camera')
            break
        annotated = frame.copy()
        detected_markers = detect_aruco_markers_robust(frame)
        for mid, m_info in detected_markers.items():
            c_pts = m_info['corners'].astype(np.int32)
            mcx, mcy = m_info['center']
            cv2.polylines(annotated, [c_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(annotated, (int(mcx), int(mcy)), 4, (0, 0, 255), -1)
            cv2.putText(annotated, f'M:{mid}', (int(mcx) - 20, int(mcy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        det_result, mask = detect_box_360_robust(frame, template_img=template_img, detected_markers=detected_markers)
        now = time.time()
        if det_result is not None:
            no_box_counter = 0
            raw_cx = det_result['cx']
            raw_cy = det_result['cy']
            head_vx = det_result['head_vx']
            head_vy = det_result['head_vy']
            box_pts = det_result['box_points']
            h_sign = det_result['heading_sign']
            arrow_end = det_result['arrow_end']
            match_score = det_result['match_score']
            logo_box_pts = det_result['logo_box_pts']
            metrics = calculate_real_box_metrics(H, box_pts)
            world_x = metrics['world_cx']
            world_y = metrics['world_cy']
            if invert_world_x:
                world_x = 2.0 * CALIB_CENTER_X - world_x
            if invert_world_y:
                world_y = 2.0 * CALIB_CENTER_Y - world_y
            real_w = metrics['width_mm']
            real_l = metrics['length_mm']
            real_area = metrics['area_cm2']
            geom_cx = float(np.mean(box_pts[:, 0]))
            geom_cy = float(np.mean(box_pts[:, 1]))
            angle_px_deg = float(np.degrees(np.arctan2(head_vy, head_vx)))
            if h_flip_y:
                world_vx = head_vx
                world_vy = -head_vy
            else:
                world_vx = head_vx
                world_vy = head_vy
            raw_world_angle = float(np.degrees(np.arctan2(world_vy, world_vx)))
            robot_angle = raw_world_angle + model_angle_offset
            if INVERT_ROBOT_ANGLE:
                robot_angle = -robot_angle
            robot_angle = (robot_angle + 180.0) % 360.0 - 180.0

            # ทิศทางของลูกศรบนระนาบพิกัดโต๊ะจริง (World Coordinates)
            dir_world_x = -world_vx if invert_world_x else world_vx
            dir_world_y = -world_vy if invert_world_y else world_vy
            rad_world = math.atan2(dir_world_y, dir_world_x)

            # เลื่อนจุดหยิบไปทางทิศลูกศร 10 mm ตามที่กำหนด
            pick_offset_along_arrow = PICK_OFFSET_ALONG_ARROW_MM
            pick_world_x = world_x + pick_offset_along_arrow * math.cos(rad_world)
            pick_world_y = world_y + pick_offset_along_arrow * math.sin(rad_world)

            # พิกัดจุดหยิบบนจอภาพ (Pixels) สำหรับวาดจุดสีน้ำเงิน
            box_len_px = float(det_result.get('box_length_px', np.linalg.norm(np.array(arrow_end) - np.array([geom_cx, geom_cy])) * 2.5))
            px_per_mm = box_len_px / max(1.0, real_l)
            offset_px = pick_offset_along_arrow * px_per_mm
            pick_px = float(geom_cx + offset_px * head_vx)
            pick_py = float(geom_cy + offset_px * head_vy)

            if smooth_world_x is None:
                smooth_world_x = world_x
                smooth_world_y = world_y
                smooth_sin_ang = np.sin(np.radians(robot_angle))
                smooth_cos_ang = np.cos(np.radians(robot_angle))
                smooth_box_pts = box_pts.astype(np.float32)
                smooth_geom_cx = geom_cx
                smooth_geom_cy = geom_cy
                smooth_real_w = real_w
                smooth_real_l = real_l
                smooth_real_area = real_area
                smooth_arrow_end = np.array(arrow_end, dtype=np.float32)
                smooth_match_score = match_score
                smooth_logo_box_pts = logo_box_pts.astype(np.float32) if logo_box_pts is not None else None
                smooth_pick_world_x = pick_world_x
                smooth_pick_world_y = pick_world_y
                smooth_pick_px = pick_px
                smooth_pick_py = pick_py
                smooth_offset_mm = pick_offset_along_arrow
            else:
                d_pos = np.hypot(world_x - smooth_world_x, world_y - smooth_world_y)
                cur_smooth_deg = float(np.degrees(np.arctan2(smooth_sin_ang, smooth_cos_ang)))
                d_ang = abs((robot_angle - cur_smooth_deg + 180.0) % 360.0 - 180.0)

                POS_DEADBAND_MM = 2.5
                ANG_DEADBAND_DEG = 1.0

                if d_pos > POS_DEADBAND_MM or d_ang > ANG_DEADBAND_DEG:
                    alpha = 0.30
                    smooth_world_x = alpha * world_x + (1.0 - alpha) * smooth_world_x
                    smooth_world_y = alpha * world_y + (1.0 - alpha) * smooth_world_y
                    rad = np.radians(robot_angle)
                    smooth_sin_ang = alpha * np.sin(rad) + (1.0 - alpha) * smooth_sin_ang
                    smooth_cos_ang = alpha * np.cos(rad) + (1.0 - alpha) * smooth_cos_ang
                    smooth_box_pts = alpha * box_pts.astype(np.float32) + (1.0 - alpha) * smooth_box_pts
                    smooth_geom_cx = alpha * geom_cx + (1.0 - alpha) * smooth_geom_cx
                    smooth_geom_cy = alpha * geom_cy + (1.0 - alpha) * smooth_geom_cy
                    smooth_real_w = alpha * real_w + (1.0 - alpha) * smooth_real_w
                    smooth_real_l = alpha * real_l + (1.0 - alpha) * smooth_real_l
                    smooth_real_area = alpha * real_area + (1.0 - alpha) * smooth_real_area
                    smooth_arrow_end = alpha * np.array(arrow_end, dtype=np.float32) + (1.0 - alpha) * smooth_arrow_end
                    smooth_match_score = alpha * match_score + (1.0 - alpha) * smooth_match_score
                    if logo_box_pts is not None:
                        if smooth_logo_box_pts is None:
                            smooth_logo_box_pts = logo_box_pts.astype(np.float32)
                        else:
                            smooth_logo_box_pts = alpha * logo_box_pts.astype(np.float32) + (1.0 - alpha) * smooth_logo_box_pts
                    smooth_pick_world_x = alpha * pick_world_x + (1.0 - alpha) * smooth_pick_world_x
                    smooth_pick_world_y = alpha * pick_world_y + (1.0 - alpha) * smooth_pick_world_y
                    smooth_pick_px = alpha * pick_px + (1.0 - alpha) * smooth_pick_px
                    smooth_pick_py = alpha * pick_py + (1.0 - alpha) * smooth_pick_py
                    smooth_offset_mm = alpha * pick_offset_along_arrow + (1.0 - alpha) * smooth_offset_mm
                else:
                    pass

            disp_world_x = float(smooth_world_x)
            disp_world_y = float(smooth_world_y)
            disp_pick_world_x = float(smooth_pick_world_x)
            disp_pick_world_y = float(smooth_pick_world_y)
            disp_pick_px = int(round(smooth_pick_px))
            disp_pick_py = int(round(smooth_pick_py))
            disp_offset_mm = float(smooth_offset_mm)
            disp_world_y = float(smooth_world_y)
            disp_robot_angle = float(np.degrees(np.arctan2(smooth_sin_ang, smooth_cos_ang)))
            disp_box_pts = smooth_box_pts.astype(np.int32)
            disp_cx = int(round(smooth_geom_cx))
            disp_cy = int(round(smooth_geom_cy))
            disp_arrow_end = (int(round(smooth_arrow_end[0])), int(round(smooth_arrow_end[1])))
            disp_w = float(smooth_real_w)
            disp_l = float(smooth_real_l)
            disp_area = float(smooth_real_area)
            disp_score = float(smooth_match_score)
            disp_logo_pts = smooth_logo_box_pts.astype(np.int32) if smooth_logo_box_pts is not None else None

            if box_obj is not None and orig_box_rot is not None:
                try:
                    rot = robomath.rotz(robomath.pi * disp_robot_angle / 180.0)
                    box_abs_pose = rot * orig_box_rot
                    box_abs_pose[0, 3] = disp_world_x + BOX_OFFSET_X
                    box_abs_pose[1, 3] = disp_world_y + BOX_OFFSET_Y
                    box_abs_pose[2, 3] = orig_box_z + BOX_OFFSET_Z
                    box_obj.setPoseAbs(box_abs_pose)
                except Exception:
                    pass
            if 'symbol_mask_orig' in det_result and det_result['symbol_mask_orig'] is not None:
                sym_m = det_result['symbol_mask_orig']
                annotated[sym_m > 0] = [0, 240, 255]
            cv2.polylines(annotated, [disp_box_pts], isClosed=True, color=(255, 0, 255), thickness=2)
            if disp_logo_pts is not None:
                cv2.polylines(annotated, [disp_logo_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(annotated, f'Head/Symbol ({int(disp_score * 100)}%)', (disp_logo_pts[0][0], max(20, disp_logo_pts[0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            # จุดกึ่งกลางเดิมของกล่อง (จุดสีแดง)
            cv2.circle(annotated, (disp_cx, disp_cy), 6, (0, 0, 255), -1)
            cv2.circle(annotated, (disp_cx, disp_cy), 2, (255, 255, 255), -1)
            cv2.putText(annotated, 'Center', (disp_cx - 20, disp_cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

            # เส้นเชื่อมโยงจากจุดกึ่งกลางแดง -> จุดหยิบใหม่น้ำเงิน
            cv2.line(annotated, (disp_cx, disp_cy), (disp_pick_px, disp_pick_py), (255, 200, 0), 2)

            # จุดหยิบใหม่ตามสูตรขยับตามแนวลูกศร (จุดสีน้ำเงิน BGR: 255, 0, 0)
            cv2.circle(annotated, (disp_pick_px, disp_pick_py), 8, (255, 0, 0), -1)
            cv2.circle(annotated, (disp_pick_px, disp_pick_py), 3, (255, 255, 255), -1)
            cv2.putText(annotated, 'NEW PICK', (disp_pick_px + 10, disp_pick_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 120, 0), 2)

            box_dim_str = f'{disp_w:.0f}x{disp_l:.0f}mm ({disp_area:.1f}cm2)'
            cv2.putText(annotated, box_dim_str, (disp_cx - 60, disp_cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.arrowedLine(annotated, (disp_cx, disp_cy), disp_arrow_end, (0, 255, 255), 3, tipLength=0.25)
            time_left = max(0.0, MOVE_COOLDOWN_SEC - (now - last_move_time))
            status_pick = 'READY TO PICK' if time_left == 0.0 else f'WAIT COOLDOWN ({time_left:.1f}s)'
            if show_hud:
                hud_overlay = annotated.copy()
                cv2.rectangle(hud_overlay, (5, 5), (460, 145), (0, 0, 0), -1)
                cv2.addWeighted(hud_overlay, 0.6, annotated, 0.4, 0, annotated)
                cv2.rectangle(annotated, (5, 5), (460, 145), (0, 255, 0), 1)
                inv_x_str = 'InvX:ON' if invert_world_x else 'InvX:OFF'
                inv_y_str = 'InvY:ON' if invert_world_y else 'InvY:OFF'
                cv2.putText(annotated, f'Center (X, Y): ({disp_world_x:.1f}, {disp_world_y:.1f}) mm [{inv_x_str} {inv_y_str}]', (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
                cv2.putText(annotated, f'NEW PICK (X, Y): ({disp_pick_world_x:.1f}, {disp_pick_world_y:.1f}) mm [Offset: {disp_offset_mm:+.1f}mm]', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 0), 2)
                cv2.putText(annotated, f'Box Size: {disp_w:.1f} x {disp_l:.1f} mm ({disp_area:.1f} cm2)', (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
                cv2.putText(annotated, f'Angle 360: {disp_robot_angle:.1f} deg | FlapOffset: {gripper_angle_offset:+.0f} deg', (15, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 2)
                cv2.putText(annotated, f'Symbol/Head: {int(disp_score * 100)}% | {status_pick}', (15, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 2)
                flip_str = 'FlipY:ON' if h_flip_y else 'FlipY:OFF'
                mask_str = 'Mask:ON' if show_mask else 'Mask:OFF'
                cv2.putText(annotated, f'[T]=Angle | [G]=Flap | [X/Y]=InvX/Y | [Q]=Quit', (15, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            if show_debug:
                dbg_x, dbg_y = (15, annotated.shape[0] - 130)
                cv2.rectangle(annotated, (dbg_x - 5, dbg_y - 18), (dbg_x + 420, dbg_y + 115), (0, 0, 0), -1)
                cv2.rectangle(annotated, (dbg_x - 5, dbg_y - 18), (dbg_x + 420, dbg_y + 115), (0, 200, 255), 1)
                cv2.putText(annotated, '--- ANGLE DEBUG ---', (dbg_x, dbg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
                cv2.putText(annotated, f'Pixel angle (arctan2 vy,vx): {angle_px_deg:.2f} deg', (dbg_x, dbg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(annotated, f"head_vx={head_vx:.3f}  head_vy={head_vy:.3f}  FlipY={('Y' if h_flip_y else 'N')}", (dbg_x, dbg_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                cv2.putText(annotated, f'raw_world_angle: {raw_world_angle:.2f} deg', (dbg_x, dbg_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                cv2.putText(annotated, f'Offset: {model_angle_offset:+.1f} deg', (dbg_x, dbg_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 150, 255), 1)
                cv2.putText(annotated, f'robot_angle (final): {disp_robot_angle:.2f} deg', (dbg_x, dbg_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 255), 2)
            if now - last_move_time > MOVE_COOLDOWN_SEC:
                print(f'\n[DETECT] Box {disp_w:.1f}x{disp_l:.1f} mm | Center ({disp_world_x:.1f}, {disp_world_y:.1f}) -> NEW PICK ({disp_pick_world_x:.1f}, {disp_pick_world_y:.1f}) mm [Offset {disp_offset_mm:+.1f}mm] | Angle {disp_robot_angle:.1f}°')
                try:
                    success = run_pick_and_place(robot, target_pick, target_prepick, prog_pick, prog_place, disp_pick_world_x, disp_pick_world_y, disp_robot_angle, gripper_angle_offset)
                    if success:
                        last_move_time = time.time()
                except Exception as e:
                    print(f'[ERROR] Robot execution error: {e}')
                    last_move_time = time.time()
        else:
            no_box_counter += 1
            if no_box_counter > 5:
                smooth_world_x = None
                smooth_world_y = None
                smooth_sin_ang = None
                smooth_cos_ang = None
                smooth_box_pts = None
                smooth_geom_cx = None
                smooth_geom_cy = None
                smooth_arrow_end = None
                smooth_logo_box_pts = None
                smooth_match_score = None
                smooth_pick_world_x = None
                smooth_pick_world_y = None
                smooth_pick_px = None
                smooth_pick_py = None
                smooth_offset_mm = None
            cv2.putText(annotated, '[NO BOX DETECTED]', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for mid, m_info in detected_markers.items():
            c_pts = m_info['corners'].astype(np.int32)
            mcx, mcy = m_info['center']
            cv2.polylines(annotated, [c_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(annotated, (int(mcx), int(mcy)), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f'M:{mid}', (int(mcx) - 18, int(mcy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow('RoboDK 360 Pick & Place System', annotated)
        if show_mask and mask is not None:
            cv2.imshow('Box Mask (Real-time)', mask)
        key = cv2.waitKey(1) & 255
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('m') or key == ord('M'):
            show_mask = not show_mask
            if not show_mask:
                try:
                    cv2.destroyWindow('Box Mask (Real-time)')
                except Exception:
                    pass
            print(f"[MASK] Mask display: {('ON' if show_mask else 'OFF')}")
        elif key == ord('t') or key == ord('T'):
            model_angle_offset = (model_angle_offset + 90.0 + 180.0) % 360.0 - 180.0
            np.save(CALIB_ANGLE_PATH, np.array([model_angle_offset], dtype=np.float32))
            print(f'[ANGLE] +90° offset = {model_angle_offset:+.1f} deg (Saved to {CALIB_ANGLE_PATH})')
        elif key == ord('['):
            model_angle_offset = (model_angle_offset + 5.0 + 180.0) % 360.0 - 180.0
            np.save(CALIB_ANGLE_PATH, np.array([model_angle_offset], dtype=np.float32))
            print(f'[ANGLE] +5° offset = {model_angle_offset:+.1f} deg (Saved to {CALIB_ANGLE_PATH})')
        elif key == ord(']'):
            model_angle_offset = (model_angle_offset - 5.0 + 180.0) % 360.0 - 180.0
            np.save(CALIB_ANGLE_PATH, np.array([model_angle_offset], dtype=np.float32))
            print(f'[ANGLE] -5° offset = {model_angle_offset:+.1f} deg (Saved to {CALIB_ANGLE_PATH})')
        elif key == ord('f') or key == ord('F'):
            h_flip_y = not h_flip_y
            print(f"[ANGLE] H_FLIP_Y = {h_flip_y} (Flip Y axis: {('ON' if h_flip_y else 'OFF')})")
        elif key == ord('g') or key == ord('G'):
            gripper_angle_offset = (gripper_angle_offset + 90.0 + 180.0) % 360.0 - 180.0
            print(f'[GRIPPER] White Flap Offset = {gripper_angle_offset:+.1f} deg')
        elif key == ord('x') or key == ord('X'):
            invert_world_x = not invert_world_x
            print(f"[COORD] Invert World X: {('ON' if invert_world_x else 'OFF')}")
        elif key == ord('y') or key == ord('Y'):
            invert_world_y = not invert_world_y
            print(f"[COORD] Invert World Y: {('ON' if invert_world_y else 'OFF')}")
        elif key == ord('d') or key == ord('D'):
            show_debug = not show_debug
            print(f"[DEBUG] Debug Overlay: {('ON' if show_debug else 'OFF')}")
        elif key == ord('h') or key == ord('H'):
            show_hud = not show_hud
            print(f"[HUD] HUD Display: {('ON' if show_hud else 'OFF')}")
        elif key == ord('s') or key == ord('S'):
            if det_result is not None and 'warped_box' in det_result:
                w_box = det_result['warped_box']
                if det_result.get('best_symbol_box_warped') is not None:
                    lx, ly, lw, lh = det_result['best_symbol_box_warped']
                    pad = 3
                    ly1 = max(0, ly - pad)
                    ly2 = min(w_box.shape[0], ly + lh + pad)
                    lx1 = max(0, lx - pad)
                    lx2 = min(w_box.shape[1], lx + lw + pad)
                    new_template_crop = w_box[ly1:ly2, lx1:lx2]
                else:
                    new_template_crop = w_box[15:100, 30:130]
                if new_template_crop.size > 0 and np.std(new_template_crop) > 15:
                    for t_path in [os.path.join(BASE_DIR, 'template_logo1.png'), os.path.join(BASE_DIR, 'template_logo2.png')]:
                        is_ok, buf = cv2.imencode('.png', new_template_crop)
                        if is_ok:
                            buf.tofile(t_path)
                    template_img = load_template_logo()
                    print(f'[TEMPLATE] Saved new template crop successfully!')
                else:
                    print(f'[WARNING] Selected area has low contrast. Template not saved.')
    cap.release()
    cv2.destroyAllWindows()
    print('[STATUS] Program terminated cleanly.')
if __name__ == '__main__':
    main()
