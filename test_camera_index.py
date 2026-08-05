# test_camera_index.py
# เปิดกล้องทีละ index เพื่อดูว่า index ไหนคือกล้อง USB ตัวจริงตามที่ OpenCV มองเห็น
# (ลำดับที่ OpenCV เห็นอาจไม่ตรงกับที่ pygrabber รายงานไว้ในบางเครื่อง)
#
# วิธีใช้: จะเปิดกล้องแต่ละ index ทีละตัว บอกด้วยว่า index อะไร
# ดูภาพแล้วกด 'n' เพื่อไปทดสอบ index ต่อไป, กด 'q' เพื่อออกทั้งหมด

import cv2

for index in range(5):
    print(f"\n[STATUS] กำลังทดสอบเปิดกล้อง Index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[INFO] Index {index}: เปิดไม่ได้ (ไม่มีกล้องที่ index นี้)")
        cap.release()
        continue

    print(f"[INFO] Index {index}: เปิดได้! กำลังแสดงภาพ -> กด 'n' ไปตัวต่อไป, กด 'q' ออกทั้งหมด")

    quit_all = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Index {index}: อ่านภาพไม่ได้")
            break

        cv2.putText(frame, f"Index {index} - กด n=ตัวต่อไป, q=ออก", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Test Camera Index", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('q'):
            quit_all = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if quit_all:
        break

print("\n[STATUS] ทดสอบเสร็จสิ้น จำ Index ของกล้อง USB ที่ถูกต้องไว้")
