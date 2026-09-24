import os
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'marker_config.json')
CALIB_HOMOGRAPHY_PATH = os.path.join(BASE_DIR, 'calib_homography.npy')
CALIB_ANGLE_PATH = os.path.join(BASE_DIR, 'calib_angle_offset.npy')
USB_CAMERA_INDEX = 0
MIN_BOX_AREA = 3000
SUPPORTED_DICTS = [cv2.aruco.DICT_4X4_1000, cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_5X5_100]
DETECTORS = []
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
    DETECTORS.append(cv2.aruco.ArucoDetector(d, p))

def load_marker_config():
    default_config = {'description': 'Configuration for ArUco Marker Calibration', 'marker_ids': [1, 2, 3, 4], 'marker_world_points': {'1': [-150.0, 350.0], '2': [150.0, 350.0], '3': [150.0, 550.0], '4': [-150.0, 550.0]}, 'marker_size_mm': 100.0}
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        return default_config
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'[WARNING] Failed to read {CONFIG_PATH}: {e} -> Using defaults')
        return default_config

def save_marker_config(config):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'[STATUS] Configuration saved to {CONFIG_PATH}')

def open_camera():
    print(f'[STATUS] Opening USB Camera Index {USB_CAMERA_INDEX}...')
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'[STATUS] Connected to Camera Index {USB_CAMERA_INDEX} ({frame.shape[1]}x{frame.shape[0]})')
            return cap
        cap.release()
    print(f'[ERROR] Failed to open Camera Index {USB_CAMERA_INDEX}')
    return None

def detect_aruco_markers_robust(frame):
    detected = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for detector in DETECTORS:
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, mid in enumerate(ids_flat):
                mid = int(mid)
                if mid not in detected:
                    pts = corners[i][0]
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    detected[mid] = {'corners': pts, 'center': (cx, cy)}
    return detected

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
    if edge1 >= edge2:
        vec = world_corners[1] - world_corners[0]
    else:
        vec = world_corners[2] - world_corners[1]
    world_angle_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return {'world_cx': world_cx, 'world_cy': world_cy, 'width_mm': float(real_width_mm), 'length_mm': float(real_length_mm), 'area_cm2': float(real_area_cm2), 'angle_deg': world_angle_deg, 'world_corners': world_corners}

def detect_box_opencv(frame, min_area=MIN_BOX_AREA):
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
        return (None, mask)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return (None, mask)
    largest = max(valid_contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps_factor in [0.02, 0.015, 0.025, 0.03, 0.035, 0.04]:
        cand = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(cand) == 4:
            approx = cand
            break
    if approx is None or len(approx) != 4:
        rect = cv2.minAreaRect(largest)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        (rect_cx, rect_cy), (rw, rh), angle = rect
    else:
        pts = approx.reshape(4, 2)
        box_pts = pts.astype(np.int32)
        rect_cx = float(np.mean(pts[:, 0]))
        rect_cy = float(np.mean(pts[:, 1]))
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[1]
        main_v = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
        angle = float(np.degrees(np.arctan2(main_v[1], main_v[0])))
    result = {'cx': float(rect_cx), 'cy': float(rect_cy), 'angle_deg': float(angle), 'box_points': box_pts, 'bbox': (bx, by, bx + bw, by + bh), 'area_px': cv2.contourArea(largest)}
    return (result, mask)

def main():
    config = load_marker_config()
    marker_ids = config.get('marker_ids', [1, 2, 3, 4])
    marker_world_pts = config.get('marker_world_points', {})
    cap = open_camera()
    if cap is None:
        return
    WINDOW_MAIN = 'ArUco Marker Calibration & Metric Box Measurement'
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    H = None
    if os.path.exists(CALIB_HOMOGRAPHY_PATH):
        try:
            H = np.load(CALIB_HOMOGRAPHY_PATH)
            print(f'[STATUS] Loaded existing Homography from {CALIB_HOMOGRAPHY_PATH}')
        except Exception:
            pass
    angle_offset = 0.0
    if os.path.exists(CALIB_ANGLE_PATH):
        try:
            val = np.load(CALIB_ANGLE_PATH)
            angle_offset = float(val[0])
            print(f'[STATUS] Loaded existing ANGLE_OFFSET = {angle_offset:.2f} deg')
        except Exception:
            pass
    print('\n' + '=' * 75)
    print('      Automatic Calibration using 4 ArUco Markers      ')
    print('=' * 75)
    print('Key controls:')
    print('  [C] : Compute & save Homography Matrix when all 4 markers are detected')
    print('  [E] : Edit real world coordinates (X, Y in mm)')
    print('  [R] : Quick rectangle coordinate setup (Width x Length in mm)')
    print('  [A] : Calibrate robot reference angle offset')
    print("  [B] : Toggle Bird's Eye View preview window")
    print('  [Q] / ESC : Quit program')
    print('=' * 75 + '\n')
    show_birds_eye = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('[WARNING] Failed to capture frame from camera')
            break
        annotated = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        detected_markers = detect_aruco_markers_robust(frame)
        found_ids = []
        for mid, m_info in detected_markers.items():
            c_pts = m_info['corners'].astype(np.int32)
            cx, cy = m_info['center']
            cv2.polylines(annotated, [c_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            w_pt = marker_world_pts.get(str(mid), None)
            if w_pt is not None:
                lbl = f'ID:{mid} ({w_pt[0]:.0f},{w_pt[1]:.0f})'
            else:
                lbl = f'ID:{mid}'
            cv2.putText(annotated, lbl, (int(cx) - 35, int(cy) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if mid in marker_ids:
                found_ids.append(mid)
        found_target_ids = [m for m in marker_ids if m in detected_markers]
        if len(found_target_ids) == 4:
            pts_poly = [detected_markers[mid]['center'] for mid in marker_ids]
            pts_poly = np.array(pts_poly, dtype=np.int32)
            cv2.polylines(annotated, [pts_poly], isClosed=True, color=(255, 255, 0), thickness=2)
        box_result, _ = detect_box_opencv(frame)
        if box_result is not None:
            box_pts = box_result['box_points']
            cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 0, 255), thickness=2)
            bcx = int(box_result['cx'])
            bcy = int(box_result['cy'])
            cv2.circle(annotated, (bcx, bcy), 5, (0, 0, 255), -1)
            if H is not None:
                metrics = calculate_real_box_metrics(H, box_pts)
                r_w = metrics['width_mm']
                r_l = metrics['length_mm']
                r_area = metrics['area_cm2']
                r_cx = metrics['world_cx']
                r_cy = metrics['world_cy']
                r_ang = metrics['angle_deg']
                dim_txt = f'{r_w:.0f} x {r_l:.0f} mm'
                area_txt = f'Area: {r_area:.1f} cm2'
                coord_txt = f'Pos: ({r_cx:.1f}, {r_cy:.1f}) mm'
                cv2.putText(annotated, dim_txt, (bcx - 50, bcy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(annotated, area_txt, (bcx - 50, bcy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated, coord_txt, (bcx - 50, bcy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.rectangle(annotated, (10, 10), (540, 140), (0, 0, 0), -1)
        border_color = (0, 255, 0) if len(found_target_ids) == 4 else (0, 0, 255)
        cv2.rectangle(annotated, (10, 10), (540, 140), border_color, 2)
        missing_ids = [m for m in marker_ids if m not in detected_markers]
        if len(found_target_ids) == 4:
            status_marker = f'Markers: Found ALL 4/4 {found_target_ids}'
            marker_color = (0, 255, 0)
        else:
            status_marker = f'Markers: Found {len(found_target_ids)}/4 | Missing: {missing_ids}'
            marker_color = (0, 165, 255)
        cv2.putText(annotated, status_marker, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, marker_color, 2)
        calib_status = 'Calibrated: ACTIVE (Ready to Use)' if H is not None else "Calibrated: NOT READY (Press 'C')"
        cv2.putText(annotated, calib_status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if H is not None else (0, 0, 255), 2)
        ang_status = f'Angle Offset: {angle_offset:.1f} deg'
        cv2.putText(annotated, ang_status, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        key_help = '[C]=Save Calib | [E]=Edit Coords | [R]=Rect | [A]=Angle | [Q]=Quit'
        cv2.putText(annotated, key_help, (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
        if len(detected_markers) == 0:
            cv2.rectangle(annotated, (10, h_frame - 40), (w_frame - 10, h_frame - 10), (0, 0, 180), -1)
            cv2.putText(annotated, 'TIP: Ensure markers have WHITE BORDER around the black box & good lighting', (20, h_frame - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(WINDOW_MAIN, annotated)
        if show_birds_eye and len(found_target_ids) == 4:
            warp_size = 500
            pts_canvas_dst = np.array([[50, 50], [warp_size - 50, 50], [warp_size - 50, warp_size - 50], [50, warp_size - 50]], dtype=np.float32)
            H_bev, _ = cv2.findHomography(np.array([detected_markers[i]['center'] for i in marker_ids], dtype=np.float32), pts_canvas_dst)
            if H_bev is not None:
                bev_img = cv2.warpPerspective(frame, H_bev, (warp_size, warp_size))
                cv2.putText(bev_img, "Bird's Eye Top-Down View", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Bird's Eye View Preview", bev_img)
        key = cv2.waitKey(1) & 255
        if key == ord('c') or key == ord('C'):
            if len(found_target_ids) < 4:
                print(f'[WARNING] Found only {len(found_target_ids)}/4 markers (Missing: {missing_ids}). All 4 required.')
                continue
            src_px = []
            dst_world = []
            for mid in marker_ids:
                src_px.append(detected_markers[mid]['center'])
                dst_world.append(marker_world_pts[str(mid)])
            src_px = np.array(src_px, dtype=np.float32)
            dst_world = np.array(dst_world, dtype=np.float32)
            H, status = cv2.findHomography(src_px, dst_world)
            np.save(CALIB_HOMOGRAPHY_PATH, H)
            print('\n' + '=' * 60)
            print('[SUCCESS] Computed Homography Matrix from ArUco Markers successfully!')
            print(f'[STATUS] Saved to {CALIB_HOMOGRAPHY_PATH}')
            print('Matrix H:')
            print(H)
            print('=' * 60 + '\n')
        elif key == ord('e') or key == ord('E'):
            print('\n--- Edit World Coordinates (RoboDK World mm) for each marker ---')
            for mid in marker_ids:
                curr = marker_world_pts.get(str(mid), [0.0, 0.0])
                try:
                    inp = input(f"Marker ID {mid} (Current X={curr[0]:.1f}, Y={curr[1]:.1f}) -> New 'X,Y' (Enter to skip): ").strip()
                    if inp:
                        parts = [float(p.strip()) for p in inp.split(',')]
                        if len(parts) == 2:
                            marker_world_pts[str(mid)] = parts
                            print(f'  -> Updated Marker {mid} to ({parts[0]:.1f}, {parts[1]:.1f})')
                except Exception as e:
                    print(f'  -> Invalid format ({e}). Skipping.')
            config['marker_world_points'] = marker_world_pts
            save_marker_config(config)
        elif key == ord('r') or key == ord('R'):
            print('\n--- Quick Rectangle Coordinates Setup ---')
            try:
                w_inp = float(input('Width between left-right markers (mm, e.g. 300): '))
                l_inp = float(input('Length between top-bottom markers (mm, e.g. 200): '))
                x_offset = float(input('Center X offset (mm, default 0): ') or '0')
                y_offset = float(input('Top Y offset (mm, e.g. 350): ') or '350')
                marker_world_pts['1'] = [x_offset - w_inp / 2.0, y_offset]
                marker_world_pts['2'] = [x_offset + w_inp / 2.0, y_offset]
                marker_world_pts['3'] = [x_offset + w_inp / 2.0, y_offset + l_inp]
                marker_world_pts['4'] = [x_offset - w_inp / 2.0, y_offset + l_inp]
                config['marker_world_points'] = marker_world_pts
                save_marker_config(config)
                print(f'[STATUS] Configured rectangle {w_inp}x{l_inp} mm successfully!')
                print(f"  ID 1: {marker_world_pts['1']}")
                print(f"  ID 2: {marker_world_pts['2']}")
                print(f"  ID 3: {marker_world_pts['3']}")
                print(f"  ID 4: {marker_world_pts['4']}")
            except Exception as e:
                print(f'[WARNING] Input error: {e}')
        elif key == ord('a') or key == ord('A'):
            if box_result is None:
                print('[WARNING] No box detected in frame. Place box and retry.')
                continue
            ang_px = box_result['angle_deg']
            try:
                robot_angle = float(input(f'  Detected angle in pixels = {ang_px:.1f} deg\n  Enter true robot angle (deg, typically 0 if aligned with X axis): '))
                angle_offset = robot_angle - ang_px
                np.save(CALIB_ANGLE_PATH, np.array([angle_offset], dtype=np.float32))
                print(f'[ANGLE] Saved ANGLE_OFFSET = {angle_offset:.2f} deg to {CALIB_ANGLE_PATH}\n')
            except Exception as e:
                print(f'[WARNING] Error: {e}')
        elif key == ord('b') or key == ord('B'):
            show_birds_eye = not show_birds_eye
            if not show_birds_eye:
                cv2.destroyWindow("Bird's Eye View Preview")
            print(f"[STATUS] Bird's Eye View Mode: {('ON' if show_birds_eye else 'OFF')}")
        elif key == ord('q') or key == ord('Q') or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('[STATUS] Calibration program closed.')
if __name__ == '__main__':
    main()
