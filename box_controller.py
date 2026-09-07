# box_controller.py

import numpy as np
from robodk import robolink, robomath


class BoxController:
    def __init__(self, box_name="BOX", offset_x=0.0, offset_y=0.0, offset_z=0.0,
                 max_angle_deg=90.0, invert_angle=False, rdk=None):
        """
        คลาสสำหรับควบคุมตำแหน่งและการหมุนของกล่องใน RoboDK
        
        :param box_name: ชื่อ Object ของกล่องใน RoboDK (default: "BOX")
        :param offset_x: ค่าชดเชยแกน X (มม.)
        :param offset_y: ค่าชดเชยแกน Y (มม.)
        :param offset_z: ค่าชดเชยแกน Z (มม.)
        :param max_angle_deg: มุมหมุนสูงสุดที่ยอมรับ (องศา)
        :param invert_angle: กลับทิศทางมุมหมุนหรือไม่
        :param rdk: robolink.Robolink instance (ถ้ามีแล้ว ส่งเข้ามาได้)
        """
        self.box_name = box_name
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.max_angle_deg = max_angle_deg
        self.invert_angle = invert_angle

        self.RDK = rdk if rdk is not None else self._connect_rdk()
        self.box_item = None
        self.orig_box_rot = None
        self.orig_box_abs = None
        self.orig_box_z = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_angle = 0.0

        self._init_box_item()

    def _connect_rdk(self):
        """เชื่อมต่อ RoboDK รองรับหลาย Port"""
        candidate_ports = [20500, 20501, 20502, 20503]
        for p in candidate_ports:
            try:
                rdk = robolink.Robolink(port=p)
                if rdk.Connected():
                    return rdk
            except Exception:
                pass
        return robolink.Robolink()

    def _init_box_item(self):
        """ค้นหาและจำค่าเริ่มต้นของ Object กล่องใน Station"""
        if not self.RDK.Connected():
            print("[BoxController] ไม่สามารถเชื่อมต่อกับ RoboDK ได้")
            return False

        self.box_item = self.RDK.Item(self.box_name, robolink.ITEM_TYPE_OBJECT)
        if not self.box_item.Valid():
            print(f"[BoxController] ไม่พบ Object '{self.box_name}' ใน RoboDK Station")
            self.box_item = None
            return False

        try:
            self.orig_box_abs = self.box_item.PoseAbs()
            self.orig_box_rot = robomath.Mat(self.orig_box_abs)
            self.orig_box_rot[0, 3] = 0
            self.orig_box_rot[1, 3] = 0
            self.orig_box_rot[2, 3] = 0
            self.orig_box_z = self.orig_box_abs[2, 3]

            self.current_x = self.orig_box_abs[0, 3]
            self.current_y = self.orig_box_abs[1, 3]
            self.current_z = self.orig_box_z
            print(f"[BoxController] พบ Object '{self.box_name}' สำเร็จ (Init Pose Z={self.orig_box_z:.1f} mm)")
            return True
        except Exception as e:
            print(f"[BoxController] ไม่สามารถอ่าน Pose เริ่มต้นของ '{self.box_name}': {e}")
            return False

    def is_valid(self):
        """ตรวจสอบว่าเชื่อมต่อกับกล่องใน RoboDK ได้หรือไม่"""
        return self.box_item is not None and self.box_item.Valid()

    def update_pose(self, world_x, world_y, angle_deg=0.0, world_z=None):
        """
        อัปเดตตำแหน่งและการหมุนของกล่องใน RoboDK แบบ Real-time
        
        :param world_x: พิกัดแกน X จริง (มม.)
        :param world_y: พิกัดแกน Y จริง (มม.)
        :param angle_deg: มุมหมุนรอบแกน Z (องศา)
        :param world_z: พิกัดแกน Z (มม.) ถ้าเป็น None จะใช้ความสูงเริ่มต้น + offset_z
        :return: bool สถานะการสั่งขยับ
        """
        if not self.is_valid():
            return False

        effective_angle = -angle_deg if self.invert_angle else angle_deg
        if self.max_angle_deg > 0:
            effective_angle = max(-self.max_angle_deg, min(self.max_angle_deg, effective_angle))

        z_target = world_z if world_z is not None else (self.orig_box_z + self.offset_z)

        try:
            rot = robomath.rotz(robomath.pi * effective_angle / 180.0)
            box_abs_pose = rot * self.orig_box_rot
            box_abs_pose[0, 3] = float(world_x) + self.offset_x
            box_abs_pose[1, 3] = float(world_y) + self.offset_y
            box_abs_pose[2, 3] = float(z_target)

            self.box_item.setPoseAbs(box_abs_pose)

            self.current_x = box_abs_pose[0, 3]
            self.current_y = box_abs_pose[1, 3]
            self.current_z = box_abs_pose[2, 3]
            self.current_angle = effective_angle
            return True
        except Exception as e:
            print(f"[BoxController] ขยับกล่องไม่สำเร็จ: {e}")
            return False

    def reset_pose(self):
        """คืนตำแหน่งและการหมุนของกล่องกลับค่าเริ่มต้น"""
        if not self.is_valid() or self.orig_box_abs is None:
            return False
        try:
            self.box_item.setPoseAbs(self.orig_box_abs)
            self.current_x = self.orig_box_abs[0, 3]
            self.current_y = self.orig_box_abs[1, 3]
            self.current_z = self.orig_box_abs[2, 3]
            self.current_angle = 0.0
            return True
        except Exception as e:
            print(f"[BoxController] รีเซ็ตตำแหน่งไม่สำเร็จ: {e}")
            return False

    def get_pose(self):
        """คืนค่าพิกัดปัจจุบัน (X, Y, Z, Angle_deg)"""
        return self.current_x, self.current_y, self.current_z, self.current_angle
