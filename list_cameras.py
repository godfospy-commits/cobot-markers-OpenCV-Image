# list_cameras.py

try:
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    devices = graph.get_input_devices()
    print("[STATUS] รายชื่อกล้องที่ Windows มองเห็น (เรียงตาม index):")
    for i, name in enumerate(devices):
        print(f"  Index {i}: {name}")
except ImportError:
    print("[WARNING] ยังไม่ได้ติดตั้ง pygrabber, กำลังติดตั้งให้อัตโนมัติไม่ได้ในสคริปต์นี้")
    print("กรุณารันคำสั่งนี้ก่อน แล้วรันสคริปต์นี้ใหม่อีกครั้ง:")
    print("  E:\\RoboDK\\Python-Embedded\\python.exe -m pip install pygrabber")
