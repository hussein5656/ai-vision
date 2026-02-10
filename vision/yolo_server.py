"""Standalone YOLO detection server (runs in subprocess).

Reads JPEG-encoded frame data from stdin:
  - 4-byte big-endian length
  - JPEG bytes
  - \n separator (optional)

Outputs JSON detections to stdout:
  - One JSON line per frame
  - {"detections": [...], "frame_id": N}

This runs in a separate process to isolate torch/YOLO from the UI.
"""

import sys
import json
import struct
import cv2
import numpy as np
import os
import time
from io import BytesIO

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"[YOLO Server] Warning: ultralytics import failed: {e}", file=sys.stderr)


DEFAULT_CONF = float(os.environ.get("YOLO_CONF", "0.35"))
BASE_IMGSZ = int(os.environ.get("YOLO_BASE_IMGSZ", "640"))
MAX_IMGSZ = int(os.environ.get("YOLO_MAX_IMGSZ", "1024"))
SHARPEN_AMOUNT = float(os.environ.get("YOLO_SHARPEN_AMOUNT", "0.25"))
AUTO_TUNE_IMGSZ = os.environ.get("YOLO_AUTO_TUNE_IMGSZ", "1") != "0"
TARGET_LATENCY = float(os.environ.get("YOLO_TARGET_LATENCY", "0.085"))
ADJUST_INTERVAL = int(os.environ.get("YOLO_TUNE_INTERVAL", "6"))
DECREASE_STEP = int(os.environ.get("YOLO_IMGSZ_DOWN", "96"))
INCREASE_STEP = int(os.environ.get("YOLO_IMGSZ_UP", "32"))
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")


def _round_up_stride(value: int, stride: int = 32) -> int:
    return int(((value + stride - 1) // stride) * stride)


def _select_imgsz(frame_shape) -> int:
    height, width = frame_shape[:2]
    longer_side = max(height, width)
    if longer_side <= BASE_IMGSZ:
        return BASE_IMGSZ
    target = _round_up_stride(longer_side)
    return min(MAX_IMGSZ, target)


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Mild sharpening to keep distant edges visible without heavy cost."""
    if SHARPEN_AMOUNT <= 0:
        return frame
    try:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        enhanced = cv2.addWeighted(frame, 1.0 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)
        return enhanced
    except Exception:
        return frame


def main():
    """Main server loop."""
    if not YOLO_AVAILABLE:
        print(json.dumps({"error": "ultralytics not available"}))
        sys.exit(1)

    try:
        model = YOLO(MODEL_PATH)
        print(json.dumps({"status": "ready"}), file=sys.stdout, flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Model load failed: {e}"}), file=sys.stdout, flush=True)
        sys.exit(1)

    frame_id = 0
    current_imgsz = BASE_IMGSZ
    smoothed_latency = TARGET_LATENCY
    frames_since_adjust = 0
    while True:
        try:
            # Read 4-byte length prefix
            len_bytes = sys.stdin.buffer.read(4)
            if not len_bytes or len(len_bytes) < 4:
                break

            jpeg_len = struct.unpack('>I', len_bytes)[0]
            if jpeg_len > 100_000_000:  # Sanity check (100 MB max)
                continue

            # Read JPEG data
            jpeg_data = sys.stdin.buffer.read(jpeg_len)
            if len(jpeg_data) < jpeg_len:
                break

            # Decode JPEG
            try:
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            # Optional enhancement keeps small/distant objects crisper
            processed_frame = _enhance_frame(frame)

            # Run detection with adaptive resolution to retain distant detail
            try:
                requested_imgsz = _select_imgsz(processed_frame.shape)
                imgsz = min(current_imgsz, requested_imgsz)
                start = time.perf_counter()
                results = model.track(
                    processed_frame,
                    persist=True,
                    conf=DEFAULT_CONF,
                    imgsz=imgsz,
                    verbose=False,
                )
                duration = time.perf_counter() - start
                smoothed_latency = (smoothed_latency * 0.9) + (duration * 0.1)
                frames_since_adjust += 1
                if AUTO_TUNE_IMGSZ and frames_since_adjust >= ADJUST_INTERVAL:
                    frames_since_adjust = 0
                    if smoothed_latency > TARGET_LATENCY * 1.25 and current_imgsz > BASE_IMGSZ:
                        current_imgsz = max(BASE_IMGSZ, current_imgsz - DECREASE_STEP)
                    elif smoothed_latency < TARGET_LATENCY * 0.7 and current_imgsz < requested_imgsz:
                        current_imgsz = min(requested_imgsz, current_imgsz + INCREASE_STEP)
            except Exception as e:
                output = json.dumps({"frame_id": frame_id, "error": str(e)})
                print(output, file=sys.stdout, flush=True)
                frame_id += 1
                continue

            # Extract detections
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        try:
                            cls = int(box.cls.item()) if hasattr(box, 'cls') else 0
                            xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else [0,0,0,0]
                            x1, y1, x2, y2 = [float(v) for v in xyxy]
                            tid = int(box.id.item()) if hasattr(box, 'id') and box.id is not None else None
                            conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                            
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            detections.append({
                                "track_id": tid,
                                "class": cls,
                                "bbox": [x1, y1, x2, y2],
                                "center": [cx, cy],
                                "confidence": conf,
                            })
                        except Exception:
                            continue

            output = json.dumps({"frame_id": frame_id, "detections": detections})
            print(output, file=sys.stdout, flush=True)
            frame_id += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(json.dumps({"error": f"Unexpected: {e}"}), file=sys.stdout, flush=True)
            break


if __name__ == "__main__":
    main()
