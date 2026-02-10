import time
from urllib.parse import urlparse, urlunparse, urljoin
import cv2
import numpy as np
from typing import Optional, Any, List

try:
    import requests
    from requests.exceptions import RequestException
    REQUESTS_AVAILABLE = True
    try:
        requests.packages.urllib3.disable_warnings()
    except Exception:
        pass
except Exception:
    requests = None  # type: ignore
    RequestException = Exception  # type: ignore
    REQUESTS_AVAILABLE = False

REMOTE_PREFIXES = ("rtsp://", "http://", "https://")


class HTTPStreamReader:
    """HTTP/HTTPS stream reader supporting MJPEG and snapshot endpoints."""

    def __init__(self, url: str, timeout: int = 5):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Le support HTTP(S) nécessite le paquet 'requests'.")
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.response: Optional[Any] = None
        self.iterator = None
        self.buffer = bytearray()
        self._closed = False
        self.mode = None  # 'mjpeg' or 'snapshot'
        self.boundary: Optional[bytes] = None
        self.verify = not self.url.lower().startswith("https://")
        self._last_snapshot_time = 0.0
        self.snapshot_interval = 0.0  # no artificial throttle; let source dictate FPS
        self.chunk_size = 4096
        self.actual_url = url
        self._connect()

    def _connect(self):
        resp = self.session.get(
            self.url,
            stream=True,
            timeout=self.timeout,
            verify=self.verify,
            headers={"User-Agent": "SurveillancePro/1.0"}
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} ({resp.reason})")

        content_type = resp.headers.get("Content-Type", "").lower()
        if "multipart/x-mixed-replace" in content_type:
            self.mode = "mjpeg"
            boundary_token = None
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary_token = part.split("=", 1)[1]
                    break
            if boundary_token:
                if not boundary_token.startswith("--"):
                    boundary_token = "--" + boundary_token
                self.boundary = boundary_token.encode('utf-8', errors='ignore')
            self.response = resp
            self.iterator = resp.iter_content(chunk_size=self.chunk_size)
        else:
            # Snapshot mode (one JPEG per request)
            resp.close()
            self.mode = "snapshot"

    def read(self) -> Optional[np.ndarray]:
        if self._closed:
            return None
        if self.mode == "mjpeg":
            return self._read_mjpeg()
        return self._read_snapshot()

    def read_first_frame(self, timeout: float = 8.0) -> Optional[np.ndarray]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.1)
        return None

    def _read_mjpeg(self) -> Optional[np.ndarray]:
        if self.iterator is None:
            return None
        frame = self._consume_latest_frame()
        if frame is not None:
            return frame
        try:
            while True:
                chunk = next(self.iterator)
                if not chunk:
                    continue
                self.buffer.extend(chunk)
                frame = self._consume_latest_frame()
                if frame is not None:
                    return frame
        except StopIteration:
            return frame
        except RequestException:
            return frame

    def _read_snapshot(self) -> Optional[np.ndarray]:
        try:
            resp = self.session.get(
                self.url,
                stream=False,
                timeout=self.timeout,
                verify=self.verify,
                headers={"User-Agent": "SurveillancePro/1.0"}
            )
            if resp.status_code >= 400:
                resp.close()
                return None
            data = resp.content
            resp.close()
            if not data:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self._last_snapshot_time = time.time()
            return frame
        except RequestException:
            return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.response:
                self.response.close()
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass

class Camera:
    def __init__(self, source=0):
        """
        source = 0 → webcam
        source = "video.mp4" → fichier vidéo
        source = "http://IP:PORT/video" → caméra IP
        """
        self.source = source
        self.cap = None
        self.http_stream: Optional[HTTPStreamReader] = None
        self._pending_http_frame = None
        self.is_remote_http = isinstance(self.source, str) and self.source.lower().startswith(("http://", "https://"))
        self._downscale_frames = self.is_remote_http
        self._target_height = 480

        self.cap = self._open_capture(self.source)
        if self.cap and self.cap.isOpened():
            self._init_from_capture()
        elif self.is_remote_http:
            self._init_http_stream()
        else:
            raise RuntimeError(
                "Impossible d'ouvrir la source {}. Vérifiez l'URL/flux et que le protocole est pris en charge.".format(self.source)
            )
        
    def _init_from_capture(self):
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.fps = self.get_fps()
        try:
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            self.frame_count = 0

    def _init_http_stream(self):
        last_error = None
        for candidate in self._build_http_candidates(self.source):
            try:
                print(f"[Camera] Tentative HTTP: {candidate}")
                stream = HTTPStreamReader(candidate)
                first = stream.read_first_frame(timeout=6.0)
                if first is None:
                    stream.close()
                    last_error = "aucune image n'a été reçue"
                    continue
                self.http_stream = stream
                self.source = candidate
                self._pending_http_frame = self._apply_downscale(first)
                self.height, self.width = self._pending_http_frame.shape[:2]
                self.fps = 18.0 if stream.mode == "snapshot" else 24.0
                self.frame_count = 0
                print(f"[Camera] Flux HTTP connecté sur {candidate} (mode={stream.mode})")
                return
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(f"Flux HTTP/S indisponible: {last_error or 'aucun flux compatible détecté'}")

    def get_fps(self) -> float:
        if self.cap:
            try:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                return fps if fps > 0 else 30.0
            except Exception:
                return 30.0
        return getattr(self, 'fps', 15.0)

    def read(self):
        """
        Lit une frame et renvoie None si fin
        """
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return self._apply_downscale(frame)
        if self.http_stream:
            if self._pending_http_frame is not None:
                frame = self._pending_http_frame
                self._pending_http_frame = None
                return frame
            frame = self.http_stream.read()
            if frame is None:
                return None
            return self._apply_downscale(frame)
        return None

    def release(self):
        if self.cap:
            self.cap.release()
        if self.http_stream:
            self.http_stream.close()

    def get_frame_count(self) -> int:
        return self.frame_count if self.cap else 0

    def get_frame_index(self) -> int:
        if self.cap:
            try:
                return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            except Exception:
                return 0
        return 0

    def set_frame_index(self, frame_idx: int):
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            except Exception:
                pass

    def _apply_downscale(self, frame):
        if not self._downscale_frames or frame is None:
            return frame
        h, w = frame.shape[:2]
        target_h = self._target_height
        if h <= target_h:
            return frame
        scale = target_h / float(h)
        new_w = max(1, int(w * scale))
        try:
            return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def _build_http_candidates(self, original_url: str) -> List[str]:
        candidates: List[str] = []
        if not original_url:
            return candidates
        seen = set()

        def add(url: str):
            if not url:
                return
            if url in seen:
                return
            seen.add(url)
            candidates.append(url)

        add(original_url)

        parsed = urlparse(original_url)
        without_query = parsed._replace(query='', fragment='')
        base_no_query = urlunparse(without_query)
        if base_no_query and base_no_query != original_url:
            add(base_no_query)

        base_for_join = base_no_query or original_url
        if not base_for_join.endswith('/'):
            base_for_join = base_for_join + '/'
        origin = ''
        if parsed.scheme and parsed.netloc:
            origin = f"{parsed.scheme}://{parsed.netloc}/"

        stream_suffixes = ['/video', '/videofeed', '/video_feed', '/mjpeg', '/mjpegstream']
        snapshot_suffixes = ['/shot.jpg', '/snapshot.jpg', '/photo.jpg', '/image.jpg', '/cam.jpg']
        for suffix in stream_suffixes + snapshot_suffixes:
            add(urljoin(base_for_join, suffix.lstrip('/')))
            if origin:
                add(urljoin(origin, suffix.lstrip('/')))

        if 'action=' not in original_url:
            sep = '&' if '?' in original_url else '?'
            add(original_url + f"{sep}action=stream")
            if base_no_query and base_no_query != original_url:
                add(base_no_query + f"?action=stream")

        return candidates

    def _consume_latest_frame(self) -> Optional[np.ndarray]:
        latest = None
        while True:
            start = self.buffer.find(b'\xff\xd8')
            if start == -1:
                if len(self.buffer) > 2_000_000:
                    del self.buffer[:-2048]
                break
            end = self.buffer.find(b'\xff\xd9', start + 2)
            if end == -1:
                if start > 0:
                    del self.buffer[:start]
                break
            frame_bytes = self.buffer[start:end + 2]
            del self.buffer[:end + 2]
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                latest = frame
        return latest

    def _open_capture(self, source):
        """Try multiple OpenCV backends for remote streams."""
        backends = [None]
        is_remote = isinstance(source, str) and source.lower().startswith(REMOTE_PREFIXES)
        if is_remote:
            ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
            gst_backend = getattr(cv2, "CAP_GSTREAMER", None)
            if ffmpeg_backend is not None:
                backends.append(ffmpeg_backend)
            if gst_backend is not None and gst_backend != ffmpeg_backend:
                backends.append(gst_backend)

        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source, backend)
            except Exception:
                continue

            if cap is not None and cap.isOpened():
                if is_remote:
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    except Exception:
                        pass
                return cap

            try:
                cap.release()
            except Exception:
                pass

        return None
