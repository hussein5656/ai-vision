from vision.camera import Camera
from vision.detector import Detector
from vision.tracker import Tracker

camera = Camera(source=0)  # webcam
detector = Detector(model_path="yolov8n.pt", conf=0.4)
tracker = Tracker()

while True:
    frame = camera.read()
    if frame is None:
        break

    dets = detector.detect(frame, count_classes={0, 2, 3, 5, 7})  # personnes + véhicules
    dets = tracker.update(dets)

    for d in dets:
        print(d)

camera.release()
