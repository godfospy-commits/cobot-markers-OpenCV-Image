import os
import sys
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USB_CAMERA_INDEX = 0

# ค่าช่วงสี LAB เริ่มต้นสำหรับกล่องกระดาษพัสดุ
DEFAULT_L_MIN = 80
DEFAULT_L_MAX = 245
DEFAULT_A_MIN = 126
DEFAULT_A_MAX = 155
DEFAULT_B_MIN = 140
DEFAULT_B_MAX = 195

def enhance_image_clahe(frame):
    """ ปรับสมดุลแสงเงาด้วย CLAHE บนช่อง L ของระบบสี LAB """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    enhanced_lab = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def draw_header(img, title, color=(0, 255, 0)):
    """ วาดแถบหัวข้อของแต่ละช่องให้สวยงามอ่านง่าย """
    out = img.copy()
    h, w = out.shape[:2]
    # แถบพื้นหลังสีดำโปร่งแสงด้านบน
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.line(out, (0, 36), (w, 36), (60, 60, 60), 1)
    cv2.putText(out, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out

def nothing(x):
    pass

def main():
    print("=" * 70)
    print("   Real-time LAB Color Object Tracking Test (Live Camera)   ")
    print("=" * 70)

    # เปิดกล้องเว็บแคม
    print(f"[STATUS] Connecting to USB Camera Index {USB_CAMERA_INDEX}...")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)

    use_camera = True
    if not cap.isOpened():
        print(f"[WARNING] Cannot open Camera Index {USB_CAMERA_INDEX}.")
        print("[STATUS] Searching for fallback images in folder...")
        use_camera = False

    # สร้างหน้าต่างหลักและหน้าต่าง Trackbar
    win_name = "Live LAB Color Object Tracking (4-Panel View)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 800)

    trackbar_win = "Adjust LAB Threshold (Sliders)"
    cv2.namedWindow(trackbar_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(trackbar_win, 450, 280)

    cv2.createTrackbar("L Min", trackbar_win, DEFAULT_L_MIN, 255, nothing)
    cv2.createTrackbar("L Max", trackbar_win, DEFAULT_L_MAX, 255, nothing)
    cv2.createTrackbar("A Min", trackbar_win, DEFAULT_A_MIN, 255, nothing)
    cv2.createTrackbar("A Max", trackbar_win, DEFAULT_A_MAX, 255, nothing)
    cv2.createTrackbar("B Min", trackbar_win, DEFAULT_B_MIN, 255, nothing)
    cv2.createTrackbar("B Max", trackbar_win, DEFAULT_B_MAX, 255, nothing)

    kernel = np.ones((5, 5), np.uint8)
    prev_time = time.time()
    fps = 0.0

    print("\n[CONTROLS]")
    print("  [Q] or [ESC] : ปิดโปรแกรม")
    print("  [S]          : บันทึกภาพ Snapshot 4 ช่องเป็นไฟล์ PNG")
    print("  [R]          : รีเซ็ตค่าสไลด์เดอร์เป็นค่าเริ่มต้น")
    print("  สไลด์เดอร์    : ปรับจูนค่า L, a, b ได้แบบ Real-time\n")

    # ภาพสำรองกรณีไม่มีกล้อง
    fallback_frame = None
    if not use_camera:
        for fname in ['IMG_1.jpg', 'IMG_2.jpg', 'sample.png', 'test.png']:
            fpath = os.path.join(BASE_DIR, fname)
            if os.path.exists(fpath):
                fallback_frame = cv2.imread(fpath)
                break
        if fallback_frame is None:
            uploaded_dir = r'C:\Users\Windows\.gemini\antigravity-ide\brain\59078c2c-4d3a-4649-8bab-b156a490e067\.user_uploaded'
            if os.path.exists(uploaded_dir):
                import glob
                imgs = glob.glob(os.path.join(uploaded_dir, '*.png'))
                if imgs:
                    imgs.sort(key=os.path.getmtime, reverse=True)
                    fallback_frame = cv2.imread(imgs[0])

    while True:
        if use_camera:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Lost camera signal.")
                break
        else:
            if fallback_frame is not None:
                frame = fallback_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera / Image Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # คำนวณ FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (now - prev_time)))
        prev_time = now

        # อ่านค่าจาก Trackbars
        l_min = cv2.getTrackbarPos("L Min", trackbar_win)
        l_max = cv2.getTrackbarPos("L Max", trackbar_win)
        a_min = cv2.getTrackbarPos("A Min", trackbar_win)
        a_max = cv2.getTrackbarPos("A Max", trackbar_win)
        b_min = cv2.getTrackbarPos("B Min", trackbar_win)
        b_max = cv2.getTrackbarPos("B Max", trackbar_win)

        # 1. ปรับแสงเงาด้วย CLAHE
        enhanced_bgr = enhance_image_clahe(frame)

        # 2. แปลงภาพเป็นระบบสี LAB
        lab_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
        lab_blur = cv2.GaussianBlur(lab_frame, (5, 5), 0)

        # 3. ตัดสีตามช่วง LAB (Color Segmentation)
        lower_bound = np.array([l_min, a_min, b_min], dtype=np.uint8)
        upper_bound = np.array([l_max, a_max, b_max], dtype=np.uint8)
        mask = cv2.inRange(lab_blur, lower_bound, upper_bound)

        # กรองสัญญาณรบกวนด้วย Morphology Close + Open
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. ตัดเฉพาะวัตถุออกมา (Bitwise AND)
        segmented = cv2.bitwise_and(frame, frame, mask=mask_clean)

        # แปลง Mask เป็นสีแบบ Viridis (เหลืองบนพื้นม่วงเข้ม เหมือนในชีท Lab)
        mask_viridis = cv2.applyColorMap(mask_clean, cv2.COLORMAP_VIRIDIS)

        # ย่อขนาดแต่ละช่องให้มีขนาดเท่ากันก่อนนำมารวม 2x2
        disp_w, disp_h = 600, 420
        p1 = cv2.resize(frame, (disp_w, disp_h))
        p2 = cv2.resize(lab_frame, (disp_w, disp_h))
        p3 = cv2.resize(mask_viridis, (disp_w, disp_h))
        p4 = cv2.resize(segmented, (disp_w, disp_h))

        # ตกแต่งหัวข้อของแต่ละช่อง
        p1 = draw_header(p1, "1. Original Image (Live)", color=(255, 255, 255))
        p2 = draw_header(p2, "2. Original Image LAB", color=(255, 200, 100))
        p3 = draw_header(p3, f"3. Mask Image (L:{l_min}-{l_max} a:{a_min}-{a_max} b:{b_min}-{b_max})", color=(0, 240, 255))
        p4 = draw_header(p4, "4. Object Tracking (Segmented Box)", color=(0, 255, 0))

        # รวมภาพ 4 ช่อง (2x2 Grid)
        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        combined = np.vstack([top_row, bot_row])

        # แถบแสดง FPS และคำแนะนำการใช้งานด้านล่าง
        banner = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"FPS: {fps:.1f} | [S]=Save Snapshot | [R]=Reset Sliders | [Q/ESC]=Quit", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        final_view = np.vstack([combined, banner])

        cv2.imshow(win_name, final_view)

        key = cv2.waitKey(1) & 255
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_path = os.path.join(BASE_DIR, 'live_lab_test_snapshot.png')
            cv2.imwrite(save_path, final_view)
            print(f"[STATUS] Saved snapshot to: {save_path}")
        elif key == ord('r') or key == ord('R'):
            cv2.setTrackbarPos("L Min", trackbar_win, DEFAULT_L_MIN)
            cv2.setTrackbarPos("L Max", trackbar_win, DEFAULT_L_MAX)
            cv2.setTrackbarPos("A Min", trackbar_win, DEFAULT_A_MIN)
            cv2.setTrackbarPos("A Max", trackbar_win, DEFAULT_A_MAX)
            cv2.setTrackbarPos("B Min", trackbar_win, DEFAULT_B_MIN)
            cv2.setTrackbarPos("B Max", trackbar_win, DEFAULT_B_MAX)
            print("[STATUS] Reset sliders to defaults.")

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()
    print("[STATUS] Program closed cleanly.")

if __name__ == '__main__':
    main()
