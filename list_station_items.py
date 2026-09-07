# list_station_items.py

from robodk import robolink

RDK = robolink.Robolink()
items = RDK.ItemList()

print("[STATUS] รายชื่อ items ทั้งหมดใน station:\n")

type_names = {
    1: "ROBOT",
    2: "FRAME",
    3: "TOOL",
    4: "OBJECT",
    5: "TARGET",
    6: "PROGRAM",
    7: "INSTRUCTION",
    8: "STATION",
    9: "ROBOT_ARM",
    10: "CAMERA",
    11: "CURVE",
    12: "POINT",
    19: "MACHINING",
}

for item in items:
    t = item.Type()
    tname = type_names.get(t, f"UNKNOWN({t})")
    print(f"  [{tname}] {item.Name()}")

print("\n[STATUS] ให้สังเกตหา item ประเภท PROGRAM ที่ชื่อเกี่ยวกับกริปเปอร์ เช่น Gripper_Open / Gripper_Close")
