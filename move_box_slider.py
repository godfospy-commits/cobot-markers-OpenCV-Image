# move_box_slider.py

import numpy as np
import cv2
from robodk import robolink, robomath

BOX_OBJECT_NAME = "BOX"

X_MIN, X_MAX = -600, 600
Y_MIN, Y_MAX = -600, 600
Z_MIN, Z_MAX = 0, 500
ANGLE_MIN, ANGLE_MAX = -180, 180

WINDOW_NAME = "RoboDK Box Manual Controller"


def nothing(x):
    pass


def connect_robodk_box(box_name=BOX_OBJECT_NAME):
    """เชื่อมต่อ RoboDK และค้นหา Object กล่อง"""
    candidate_ports = [20500, 20501, 20502, 20503]
    RDK = None
    box_obj = None

    for port in candidate_ports:
        try:
            cand_rdk = robolink.Robolink(port=port)
            if cand_rdk.Connected():
                cand_box = cand_rdk.Item(box_name, robolink.ITEM_TYPE_OBJECT)
                if cand_box.Valid():
                    RDK = cand_rdk
                    box_obj = cand_box
                    print(f"[STATUS] เชื่อมต่อ RoboDK (Port {port}) พบ Object '{box_name}'")
                    break
        except Exception:
            pass

    if RDK is None or box_obj is None:
        RDK = robolink.Robolink()
        box_obj = RDK.Item(box_name, robolink.ITEM_TYPE_OBJECT)

    if not box_obj.Valid():
        raise RuntimeError(f"ไม่พบ Object '{box_name}' ใน RoboDK Station กรุณาเปิด RoboDK ก่อน")

    init_box_abs = box_obj.PoseAbs()
    orig_box_rot = robomath.Mat(init_box_abs)
    orig_box_rot[0, 3] = 0
    orig_box_rot[1, 3] = 0
    orig_box_rot[2, 3] = 0
    orig_box_z = float(init_box_abs[2, 3])

    return RDK, box_obj, orig_box_rot, orig_box_z, init_box_abs


def main():
    print("=" * 60)
    print("      RoboDK Manual Box Controller (Trackbar Slider GUI)      ")
    print("=" * 60)

    try:
        RDK, box_obj, orig_box_rot, orig_box_z, init_box_abs = connect_robodk_box()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    init_x = int(init_box_abs[0, 3])
    init_y = int(init_box_abs[1, 3])
    init_z = int(orig_box_z)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("Pos X (mm)", WINDOW_NAME, init_x - X_MIN, X_MAX - X_MIN, nothing)
    cv2.createTrackbar("Pos Y (mm)", WINDOW_NAME, init_y - Y_MIN, Y_MAX - Y_MIN, nothing)
    cv2.createTrackbar("Pos Z (mm)", WINDOW_NAME, init_z - Z_MIN, Z_MAX - Z_MIN, nothing)
    cv2.createTrackbar("Angle (deg)", WINDOW_NAME, 0 - ANGLE_MIN, ANGLE_MAX - ANGLE_MIN, nothing)

    print("\n[INFO] เลื่อน Trackbar ในหน้าต่าง OpenCV เพื่อขยับกล่องใน RoboDK ได้แบบ Real-Time")
    print("       กด 'r' เพื่อรีเซ็ตกลับค่าเดิม, กด 'q' หรือ ESC เพื่อออก\n")

    canvas = np.zeros((260, 500, 3), dtype=np.uint8)

    while True:
        pos_x = cv2.getTrackbarPos("Pos X (mm)", WINDOW_NAME) + X_MIN
        pos_y = cv2.getTrackbarPos("Pos Y (mm)", WINDOW_NAME) + Y_MIN
        pos_z = cv2.getTrackbarPos("Pos Z (mm)", WINDOW_NAME) + Z_MIN
        angle = cv2.getTrackbarPos("Angle (deg)", WINDOW_NAME) + ANGLE_MIN

        try:
            rot = robomath.rotz(robomath.pi * angle / 180.0)
            box_abs_pose = rot * orig_box_rot
            box_abs_pose[0, 3] = float(pos_x)
            box_abs_pose[1, 3] = float(pos_y)
            box_abs_pose[2, 3] = float(pos_z)
            box_obj.setPoseAbs(box_abs_pose)
        except Exception as e:
            print(f"[WARNING] เกิดข้อผิดพลาดในการตั้งค่า Pose: {e}")

        canvas[:] = (30, 30, 30)
        cv2.putText(canvas, "RoboDK Manual Box Controller", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"Target Object : {BOX_OBJECT_NAME}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, f"X : {pos_x:+.1f} mm", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas, f"Y : {pos_y:+.1f} mm", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas, f"Z : {pos_z:+.1f} mm", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas, f"Angle : {angle:+.1f} deg", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        cv2.putText(canvas, "[R]=Reset, [Q]/ESC=Exit", (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos("Pos X (mm)", WINDOW_NAME, init_x - X_MIN)
            cv2.setTrackbarPos("Pos Y (mm)", WINDOW_NAME, init_y - Y_MIN)
            cv2.setTrackbarPos("Pos Z (mm)", WINDOW_NAME, init_z - Z_MIN)
            cv2.setTrackbarPos("Angle (deg)", WINDOW_NAME, 0 - ANGLE_MIN)
            box_obj.setPoseAbs(init_box_abs)
            print("[STATUS] รีเซ็ตกลับค่าตำแหน่งเริ่มต้น")

    cv2.destroyAllWindows()
    print("[STATUS] ปิดโปรแกรมเรียบร้อย")


if __name__ == "__main__":
    main()
