"""Client to communicate with YOLO server (subprocess).

Sends JPEG-encoded frames to the server and receives JSON detections.
"""

import json
import struct
import subprocess
import threading
import time
import cv2
import sys
import os
from typing import Optional, List, Dict, Any


class YOLOClient:
    """Client for communicating with YOLO server subprocess."""

    def __init__(self, model_path: str = "yolov8n.pt", timeout: int = 5):
        self.model_path = model_path
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self._started = False
        self._error = None
        self.server_env = self._build_server_env()
        self._start_server()

    def _start_server(self):
        """Start the YOLO server subprocess."""
        try:
            server_script = os.path.join(
                os.path.dirname(__file__), "yolo_server.py"
            )
            self.process = subprocess.Popen(
                [sys.executable, server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                preexec_fn=None,  # Windows doesn't support preexec_fn
                env=self.server_env,
            )

            # Read startup message
            try:
                line = self.process.stdout.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    msg = json.loads(line)
                    if msg.get("status") == "ready":
                        self._started = True
                        print("[YOLOClient] Server started OK")
                    else:
                        self._error = msg.get("error", "Unknown startup error")
                        print(f"[YOLOClient] Server error: {self._error}")
            except Exception as e:
                self._error = f"Startup timeout/error: {e}"
                print(f"[YOLOClient] {self._error}")
                if self.process:
                    self.process.terminate()
                    self.process = None

        except Exception as e:
            self._error = f"Failed to start server: {e}"
            print(f"[YOLOClient] {self._error}")

    def _build_server_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.model_path:
            env["YOLO_MODEL_PATH"] = self.model_path
        return env

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Send frame to server and get detections.

        Args:
            frame: OpenCV BGR frame (numpy array)

        Returns:
            List of detection dicts with track_id, class, bbox, center, confidence
        """
        if not self._started or self.process is None:
            return []

        try:
            # Encode frame as JPEG
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                return []

            jpeg_data = buf.tobytes()
            jpeg_len = len(jpeg_data)

            # Send length + data
            with self.lock:
                try:
                    self.process.stdin.write(struct.pack('>I', jpeg_len))
                    self.process.stdin.write(jpeg_data)
                    self.process.stdin.flush()
                except Exception as e:
                    print(f"[YOLOClient] Send error: {e}")
                    return []

                # Read response (with timeout)
                line = self.process.stdout.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                return []

            try:
                response = json.loads(line)
                detections = response.get("detections", [])
                return detections
            except json.JSONDecodeError:
                return []

        except Exception as e:
            print(f"[YOLOClient] detect() error: {e}")
            return []

    def is_ready(self) -> bool:
        """Check if server is running and ready."""
        return self._started and self.process is not None

    def shutdown(self):
        """Shutdown the server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self._started = False

    def __del__(self):
        self.shutdown()
