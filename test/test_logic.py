import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
# test_read_video.py
import cv2, sys
path = r"C:\itlocal\examin_finale_ia\video4.mp4"  # ajustez si nécessaire

cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("OPEN ERROR: cannot open", path)
    sys.exit(1)

print("Opened OK. FPS:", cap.get(cv2.CAP_PROP_FPS), "FRAME_COUNT:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(5):
    try:
        ret, frame = cap.read()
    except Exception as e:
        print("EXCEPTION during read:", e)
        sys.exit(2)
    if not ret:
        print("EOF or read failed at frame", i)
        break
    print("Read frame", i, "shape:", None if frame is None else frame.shape)
cap.release()
print("Done")