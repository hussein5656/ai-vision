"""Professional video thread for real-time processing."""

import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QImage
from typing import Dict, Optional, List
import cv2

from vision.camera import Camera
from vision.detector import (
    Detector,
    get_class_name,
    is_vehicle_class,
    VEHICLE_CLASS_IDS,
)
from logic.engine import LogicEngine


class ProfessionalVideoThread(QThread):
    """Worker thread for professional video processing."""
    
    frame_ready = pyqtSignal(int, bytes)
    stats_ready = pyqtSignal(int, dict)
    alert_triggered = pyqtSignal(int, str, str)
    error_occurred = pyqtSignal(int, str)
    
    def __init__(self, feed_id: int, camera_id, zones_config: Dict = None, source_type: int = 0,
                 model_path: str = "yolov8n.pt"):
        super().__init__()
        self.feed_id = feed_id
        self.camera_id = camera_id
        self.zones_config = zones_config or {}
        self.source_type = source_type
        self.is_video_source = (self.source_type == 2)
        
        self._running = False
        self._lock = QMutex()
        self._current_frame = None
        self._confidence = 0.4
        self._show_accuracy = True
        self._show_boxes = True
        self._show_labels = True
        self._show_zones = True
        self._show_centers = False
        self._playback_speed = 1.0
        self._seek_request = None
        self._playback_fraction = 0.0
        self._model_path = model_path or "yolov8n.pt"
        self._pending_model_path = None
        
        # Video processing (caméra + moteur d'inférence)
        self.camera = None
        self.detector = Detector(model_path=self._model_path, conf=self._confidence)
        self.engine = LogicEngine(rules=[])
        self.last_event = None
        
        # Statistics
        self.fps_counter = 0
        self.total_fps = 0
        self.frame_count = 0
        self.frame_total = 0
        self.current_frame_index = 0
        self.source_fps = 0
        self.current_detections = 0
        self.entries = 0
        self.exits = 0
        self.parking_violations = 0
        self.unauthorized_zones = 0
        self.total_alerts = 0
        self.is_remote_http_source = False
        
        # Tracking
        self.last_frame_time = time.time()
        self.zone_presence = {'parking': set(), 'forbidden': set()}
        self._cached_polygons = {
            'parking_zones': [],
            'forbidden_zones': [],
            'loitering_zones': []
        }
        self.hidden_classes = set()
        self.show_info_panel = True
        self.info_panel_fields = {
            'fps': True,
            'objects': True,
            'entries': True,
            'exits': True,
            'parking': True,
            'forbidden': True
        }
        self._cache_polygons()
        self._processing_mode = "background"
        self._active_max_side = 960
        self._background_max_side = 640
        self._current_max_side = self._active_max_side
        self._active_stride = 1
        self._background_stride = 2
        self._processing_stride = 1
        self._frame_cycle = 0
        self._jpeg_quality_active = 80
        self._jpeg_quality_background = 60
        self._current_jpeg_quality = self._jpeg_quality_active
    
    def _setup_camera(self):
        """Setup camera."""
        try:
            print(f"[VideoThread] Opening camera: {self.camera_id}")
            self.camera = Camera(self.camera_id)
            print(f"[VideoThread] Camera opened successfully")
            self.frame_total = self.camera.get_frame_count() if self.is_video_source else 0
            self.source_fps = self.camera.get_fps() or 0
            self.is_remote_http_source = getattr(self.camera, 'is_remote_http', False)
        except Exception as e:
            error_msg = f"Camera error: {e}"
            print(f"[VideoThread] {error_msg}")
            self.error_occurred.emit(self.feed_id, error_msg)
            self.camera = None
    
    def run(self):
        """Main processing loop."""
        # Open the camera inside the thread to ensure capture is created and used on the same thread
        self._setup_camera()
        if not self.camera:
            print("[VideoThread] Camera not available, aborting run()")
            return

        self._running = True
        self.frame_count = 0
        frame_skip = 0
        last_frame_log = 0
        
        while self._running:
            try:
            # Handle pending seek requests before reading a frame
                self._lock.lock()
                seek_request = self._seek_request
                self._seek_request = None
                playback_speed = self._playback_speed
                self._lock.unlock()

                if seek_request is not None and self.is_video_source and self.camera:
                    if self.frame_total <= 0:
                        self.frame_total = self.camera.get_frame_count()
                    target_frame = 0
                    if self.frame_total > 0:
                        target_frame = int(seek_request * max(1, self.frame_total - 1))
                    self.camera.set_frame_index(target_frame)
                    self.current_frame_index = target_frame

                start_time = time.time()
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self._frame_cycle += 1
                self._lock.lock()
                stride = self._processing_stride
                max_side = self._current_max_side
                jpeg_quality = self._current_jpeg_quality
                self._lock.unlock()
                if stride > 1 and (self._frame_cycle % stride) != 0:
                    continue
                
                # Skip frames for live cameras, process all for videos
                is_video = isinstance(self.camera_id, str)
                if not is_video and frame_skip % 2 != 0:
                    frame_skip += 1
                    continue
                frame_skip += 1
                
                # Get confidence threshold
                self._lock.lock()
                conf = self._confidence
                pending_model = self._pending_model_path
                self._pending_model_path = None
                self._lock.unlock()

                if pending_model and pending_model != self._model_path:
                    self._reload_detector(pending_model)
                    if self.detector is None:
                        time.sleep(0.1)
                        continue
                
                # Detection (YOLO provides native tracking via track_id)
                tracks = self.detector.detect(frame, conf=conf, max_side=max_side) if self.detector else []
                self.current_detections = len(tracks)
                self._update_zone_presence(tracks)
                if self.is_video_source:
                    self.current_frame_index = self.camera.get_frame_index()
                
                # Event processing
                events = self.engine.process_frame(tracks)
                self._process_events(events)
                
                # Log every 30 frames to show we're processing
                if self.frame_count - last_frame_log >= 30:
                    print(f"[VideoThread] Processed {self.frame_count} frames, detections: {len(tracks)}, events: {len(events)}")
                    last_frame_log = self.frame_count
                
                # Store frame for zone editor
                self._lock.lock()
                self._current_frame = frame.copy()
                self._lock.unlock()
                
                # Visualization
                vis_frame = self._draw_professional_viz(frame, tracks)
                
                # Convert and emit
                self.frame_count += 1
                self._update_fps()
                self._emit_frame(vis_frame, jpeg_quality)

                # Throttle to video FPS for smooth, real-time-like playback
                if not self.is_remote_http_source:
                    try:
                        fps = self.source_fps or self.camera.get_fps()
                        if fps:
                            self.source_fps = fps
                        if not fps or fps <= 0:
                            fps = 30.0
                        target = 1.0 / fps
                        adjusted = target / max(0.1, playback_speed)
                        elapsed = time.time() - start_time
                        to_sleep = adjusted - elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                    except Exception:
                        pass

                # Skip extra frames to speed up playback on recorded videos
                if self.is_video_source and playback_speed > 1.0:
                    skip_frames = int(playback_speed) - 1
                    fractional = playback_speed - int(playback_speed)
                    if fractional > 0:
                        self._playback_fraction += fractional
                        if self._playback_fraction >= 1.0:
                            skip_frames += 1
                            self._playback_fraction -= 1.0
                    for _ in range(max(0, skip_frames)):
                        skipped = self.camera.read()
                        if skipped is None:
                            break
                        self.frame_count += 1
                        self.current_frame_index = self.camera.get_frame_index()
                else:
                    self._playback_fraction = 0.0
                
                # Emit stats periodically
                if self.frame_count % 10 == 0:
                    self._emit_stats()
                
            except Exception as e:
                import traceback, datetime
                tb = traceback.format_exc()
                error_msg = f"Processing error: {type(e).__name__}: {e}\n{tb}"
                print(f"[VideoThread] {error_msg}")
                # Save trace to log file for post-mortem
                try:
                    with open('app_errors.log', 'a', encoding='utf-8') as fh:
                        fh.write(f"[{datetime.datetime.now().isoformat()}] {error_msg}\n")
                except Exception:
                    pass
                # Emit abbreviated error message for UI
                short_msg = f"Processing error: {type(e).__name__}: {e}"
                self.error_occurred.emit(self.feed_id, short_msg)
                break
    
    def _draw_professional_viz(self, frame, tracks):
        """Draw professional visualization."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Draw bounding boxes and IDs
        for track in tracks:
            if self._should_hide_track(track):
                continue
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_label = self._get_track_label(track)
            label_color = (0, 200, 255) if self._is_vehicle_track(track) else (200, 200, 200)
            
            if self._show_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if self._show_labels:
                label_y = max(y1 - 20, 15)
                cv2.putText(vis, class_label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 2)
                cv2.putText(vis, f"ID:{track_id}", (x1, label_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            # Draw confidence if enabled
            if self._show_accuracy and 'confidence' in track:
                conf = int(track['confidence'] * 100)
                cv2.putText(vis, f"{conf}%", (x2-50, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

            if self._show_centers and 'center' in track:
                cx, cy = track['center']
                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        
        # Draw zones
        if self._show_zones:
            self._draw_zones(vis, self.zones_config)
        
        # Draw info panel
        self._draw_info_panel(vis)
        
        return vis
    
    def _draw_zones(self, frame, zones_config):
        """Draw zone overlays."""
        h, w = frame.shape[:2]
        
        # Counting lines
        for line in zones_config.get('counting_lines', []):
            direction = line.get('direction', 'horizontal')
            if direction == 'vertical':
                x = int(line.get('line_y', w // 2))
                start_y = int(line.get('y_start', 0))
                end_y = line.get('y_end', h)
                if end_y is None:
                    end_y = h
                try:
                    end_y = int(end_y)
                except Exception:
                    end_y = h
                if start_y >= end_y:
                    start_y, end_y = 0, h
                start_y = max(0, min(start_y, h))
                end_y = max(0, min(end_y, h))
                cv2.line(frame, (x, start_y), (x, end_y), (0, 255, 255), 3)
                label_y = max(20, start_y + 15)
                cv2.putText(frame, "COUNTING", (max(10, x - 40), label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                y = int(line.get('line_y', h // 2))
                start_x = int(line.get('x_start', 0))
                end_x = line.get('x_end', w)
                if end_x is None:
                    end_x = w
                try:
                    end_x = int(end_x)
                except Exception:
                    end_x = w
                if start_x >= end_x:
                    start_x, end_x = 0, w
                start_x = max(0, min(start_x, w))
                end_x = max(0, min(end_x, w))
                cv2.line(frame, (start_x, y), (end_x, y), (0, 255, 255), 3)
                label_x = max(10, start_x + 5)
                cv2.putText(frame, "COUNTING", (label_x, max(20, y-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Forbidden zones
        forbidden_polys = self._cached_polygons.get('forbidden_zones', [])
        for pts in forbidden_polys:
            if pts is not None:
                contour = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], True, (0, 0, 255), 2)
        
        # Parking zones
        parking_polys = self._cached_polygons.get('parking_zones', [])
        for pts in parking_polys:
            if pts is not None:
                contour = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], True, (255, 165, 0), 2)

        # Loitering zones (if configured)
        loiter_polys = self._cached_polygons.get('loitering_zones', [])
        for pts in loiter_polys:
            if pts is not None:
                contour = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], True, (255, 0, 255), 2)

    def _normalize_polygon(self, zone):
        """Convert stored zone points to a NumPy array or return None if invalid."""
        if not isinstance(zone, (list, tuple)):
            return None

        normalized = []
        for pt in zone:
            if isinstance(pt, dict):
                x = pt.get('x')
                y = pt.get('y')
                if x is None or y is None:
                    continue
                normalized.append([int(x), int(y)])
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                normalized.append([int(pt[0]), int(pt[1])])

        if len(normalized) < 3:
            return None

        try:
            return np.array(normalized, dtype=np.int32)
        except Exception:
            return None

    def _cache_polygons(self):
        """Precompute polygon arrays for each zone type."""
        cached = {
            'parking_zones': [],
            'forbidden_zones': [],
            'loitering_zones': []
        }
        for key in cached.keys():
            for zone in self.zones_config.get(key, []):
                cached[key].append(self._normalize_polygon(zone))
        self._cached_polygons = cached

    def set_processing_priority(self, mode: str):
        target = "active" if mode == "active" else "background"
        self._lock.lock()
        self._processing_mode = target
        self._processing_stride = self._active_stride if target == "active" else self._background_stride
        self._current_max_side = self._active_max_side if target == "active" else self._background_max_side
        self._current_jpeg_quality = self._jpeg_quality_active if target == "active" else self._jpeg_quality_background
        self._lock.unlock()

    def _point_in_polygon(self, polygon, x, y) -> bool:
        if polygon is None:
            return False
        contour = polygon.reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def _update_zone_presence(self, tracks):
        parking_ids = set()
        forbidden_ids = set()

        parking_polys = self._cached_polygons.get('parking_zones', [])
        forbidden_polys = self._cached_polygons.get('forbidden_zones', [])

        for det in tracks:
            center = det.get('center')
            if not center:
                continue
            cx, cy = center
            tid = det.get('track_id')

            if self._is_vehicle_track(det):
                for pts in parking_polys:
                    if pts is not None and self._point_in_polygon(pts, cx, cy):
                        parking_ids.add(tid)
                        break

            for pts in forbidden_polys:
                if pts is not None and self._point_in_polygon(pts, cx, cy):
                    forbidden_ids.add(tid)
                    break

        self.zone_presence['parking'] = parking_ids
        self.zone_presence['forbidden'] = forbidden_ids

    def _is_vehicle_track(self, det: Dict) -> bool:
        cls_id = det.get('class')
        return is_vehicle_class(cls_id)

    def _should_hide_track(self, det: Dict) -> bool:
        label = get_class_name(det.get('class')).lower()
        return label in self.hidden_classes

    def _get_track_label(self, det: Dict) -> str:
        label = get_class_name(det.get('class'))
        return label.capitalize()
    
    def _draw_info_panel(self, frame):
        """Draw info panel."""
        if not self.show_info_panel:
            return
        h, w = frame.shape[:2]

        lines = []
        if self.info_panel_fields.get('fps', True):
            lines.append((f"FPS: {self.total_fps:.1f}", (0, 255, 0)))
        if self.info_panel_fields.get('objects', True):
            lines.append((f"Objets: {self.current_detections}", (0, 255, 0)))
        if self.info_panel_fields.get('entries', True):
            lines.append((f"Entrées: {self.entries}", (0, 255, 0)))
        if self.info_panel_fields.get('exits', True):
            lines.append((f"Sorties: {self.exits}", (0, 255, 0)))
        if self.info_panel_fields.get('parking', True):
            parking_count = len(self.zone_presence.get('parking', []))
            lines.append((f"Parking: {parking_count}", (0, 200, 255)))
        if self.info_panel_fields.get('forbidden', True):
            forbidden_count = len(self.zone_presence.get('forbidden', []))
            lines.append((f"Interdit: {forbidden_count}", (0, 0, 255)))

        if not lines:
            return

        panel_height = 20 + len(lines) * 25
        cv2.rectangle(frame, (10, 10), (310, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (310, 10 + panel_height), (0, 255, 0), 2)

        y_pos = 35
        for text, color in lines:
            cv2.putText(frame, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
    
    def _process_events(self, events):
        """Process events from logic engine."""
        for event in events:
            etype = getattr(event, 'event_type', str(event))
            # Create a readable message and track counts
            if etype == 'line_in':
                self.entries += 1
                alert_type = "ENTRY"
                self.total_alerts += 1

            elif etype == 'line_out':
                self.exits += 1
                alert_type = "EXIT"
                self.total_alerts += 1

            elif etype == 'zone_time_exceeded':
                self.parking_violations += 1
                alert_type = "PARKING"
                self.total_alerts += 1

            elif etype == 'zone_enter':
                alert_type = "PARKING"
                self.total_alerts += 1

            elif etype == 'zone_exit':
                alert_type = "PARKING"
                self.total_alerts += 1

            elif etype == 'loitering':
                alert_type = "LOITERING"
                self.total_alerts += 1

            elif etype == 'anomaly':
                alert_type = "ANOMALY"
                self.total_alerts += 1
                details = getattr(event, 'details', {}) or {}
                if details.get('reason') == 'forbidden_zone':
                    self.unauthorized_zones += 1
                    alert_type = "FORBIDDEN"

            else:
                alert_type = "INFO"

            msg = f"{etype.upper()} ID:{event.track_id}"
            if getattr(event, 'details', None):
                msg += f" | {event.details}"

            # Save last event for UI/debug
            self.last_event = event.to_dict() if hasattr(event, 'to_dict') else {'type': etype, 'track_id': event.track_id}

            # Emit alert
            try:
                self.alert_triggered.emit(self.feed_id, alert_type, msg)
            except Exception:
                pass
    
    def _update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            self.fps_counter = 1.0 / delta
            self.total_fps = (self.total_fps * 0.8 + self.fps_counter * 0.2)
        self.last_frame_time = current_time
    
    def _emit_frame(self, frame, jpeg_quality: int):
        """Convert OpenCV frame to QImage and emit."""
        try:
            quality = max(30, min(95, jpeg_quality))
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret:
                return
            data = buf.tobytes()
            self.frame_ready.emit(self.feed_id, data)
        except Exception as e:
            print(f"[VideoThread] Error emitting frame: {e}")
    
    def _emit_stats(self):
        """Emit statistics."""
        stats = {
            'fps': self.total_fps,
            'current_objects': self.current_detections,
            'entries': self.entries,
            'exits': self.exits,
            'parking_violations': self.parking_violations,
            'unauthorized_zones': self.unauthorized_zones,
            'alerts': self.total_alerts,
            'parking_objects': len(self.zone_presence.get('parking', set())),
            'forbidden_objects': len(self.zone_presence.get('forbidden', set())),
            'frame_index': self.current_frame_index,
            'frame_total': self.frame_total,
            'source_fps': self.source_fps
        }
        self.stats_ready.emit(self.feed_id, stats)
    
    def get_current_frame(self):
        """Get current frame for zone editor."""
        self._lock.lock()
        frame = self._current_frame.copy() if self._current_frame is not None else None
        self._lock.unlock()
        return frame

    def set_playback_speed(self, speed: float):
        self._lock.lock()
        self._playback_speed = max(0.25, min(4.0, speed))
        self._lock.unlock()

    def seek_to_progress(self, progress: float):
        if not self.is_video_source:
            return
        progress = max(0.0, min(1.0, progress))
        self._lock.lock()
        self._seek_request = progress
        self._lock.unlock()
    
    def update_zones(self, zones_config: Dict, current_detections: List = None):
        """Update zones configuration and build rules from it.
        
        Args:
            zones_config: Dict with zone definitions
            current_detections: Optional list of current detections to initialize zone rules
        """
        self.zones_config = (zones_config or {}).copy()
        self._cache_polygons()
        print(f"\n[VideoThread] Updating zones, config={self.zones_config}")
        
        # Build logical rules based on zones configuration
        rules = []
        try:
            from logic.line_crossing import LineCrossingRule
            from logic.zone_time import ZoneTimeRule
            from logic.loitering import LoiteringRule
            from logic.forbidden_zone import ForbiddenZoneRule
        except Exception as e:
            print(f"[VideoThread] Failed to import rules: {e}")
            LineCrossingRule = None
            ZoneTimeRule = None
            LoiteringRule = None
            ForbiddenZoneRule = None

        # Counting lines
        counting_lines = self.zones_config.get('counting_lines', [])
        print(f"[VideoThread] Found {len(counting_lines)} counting lines")
        for line in counting_lines:
            if LineCrossingRule is None:
                break
            line_y = int(line.get('line_y', 0))
            direction = line.get('direction', 'horizontal')
            if direction == 'both':
                direction = 'horizontal'
            if direction == 'vertical':
                x_start = 0
                x_end = None
                y_start = int(line.get('y_start', 0))
                y_end = line.get('y_end')
            else:
                x_start = int(line.get('x_start', 0))
                x_end = line.get('x_end')
                y_start = 0
                y_end = None
            if x_end is not None:
                try:
                    x_end = int(x_end)
                except Exception:
                    x_end = None
            if y_end is not None:
                try:
                    y_end = int(y_end)
                except Exception:
                    y_end = None
            try:
                rule = LineCrossingRule(
                    line_y=line_y,
                    direction=direction,
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                )
                print(f"[VideoThread] Created LineCrossingRule: line_y={line_y}, direction={direction}")
                rules.append(rule)
            except Exception as e:
                print(f"[VideoThread] Failed to create LineCrossingRule: {e}")

        # Parking zones -> create ZoneTimeRule with default threshold if not provided
        parking_polys = self._cached_polygons.get('parking_zones', [])
        print(f"[VideoThread] Found {len(parking_polys)} parking zones")
        for i, pts in enumerate(parking_polys):
            if ZoneTimeRule is None:
                break
            if pts is None:
                continue
            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            threshold = 30.0
            try:
                rule = ZoneTimeRule(
                    zone_id=f"parking_{i}",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    time_threshold=threshold,
                    allowed_classes=VEHICLE_CLASS_IDS,
                )
                # Initialize with current detections to count already-parked vehicles
                if current_detections:
                    rule.init_with_detections(current_detections)
                print(f"[VideoThread] Created ZoneTimeRule: parking_{i}, bounds=({x1},{y1})-({x2},{y2}), threshold={threshold}s")
                rules.append(rule)
            except Exception as e:
                print(f"[VideoThread] Failed to create ZoneTimeRule for parking_{i}: {e}")

        # Loitering zones
        loitering_polys = self._cached_polygons.get('loitering_zones', [])
        print(f"[VideoThread] Found {len(loitering_polys)} loitering zones")
        for i, pts in enumerate(loitering_polys):
            if LoiteringRule is None:
                break
            if pts is None:
                continue
            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            try:
                rule = LoiteringRule(zone_id=f"loiter_{i}", x1=x1, y1=y1, x2=x2, y2=y2, time_threshold=20.0)
                print(f"[VideoThread] Created LoiteringRule: loiter_{i}, bounds=({x1},{y1})-({x2},{y2})")
                rules.append(rule)
            except Exception as e:
                print(f"[VideoThread] Failed to create LoiteringRule for loiter_{i}: {e}")

        forbidden_polys = self._cached_polygons.get('forbidden_zones', [])
        print(f"[VideoThread] Found {len(forbidden_polys)} forbidden zones")
        for i, pts in enumerate(forbidden_polys):
            if ForbiddenZoneRule is None:
                break
            if pts is None:
                continue
            try:
                rule = ForbiddenZoneRule(zone_id=f"forbidden_{i}", polygon_points=pts.tolist())
                print(f"[VideoThread] Created ForbiddenZoneRule: forbidden_{i}")
                rules.append(rule)
            except Exception as e:
                print(f"[VideoThread] Failed to create ForbiddenZoneRule for forbidden_{i}: {e}")

        self.engine = LogicEngine(rules=rules)
        print(f"[VideoThread] Total rules created: {len(rules)}\n")
    
    def set_accuracy(self, value: float):
        """Set confidence threshold."""
        self._lock.lock()
        self._confidence = value
        self._lock.unlock()
    
    def set_show_accuracy(self, show: bool):
        """Set show accuracy flag."""
        self._lock.lock()
        self._show_accuracy = show
        self._lock.unlock()

    def set_show_boxes(self, show: bool):
        self._lock.lock()
        self._show_boxes = show
        self._lock.unlock()

    def set_show_labels(self, show: bool):
        self._lock.lock()
        self._show_labels = show
        self._lock.unlock()

    def set_show_zones(self, show: bool):
        self._lock.lock()
        self._show_zones = show
        self._lock.unlock()

    def set_show_centers(self, show: bool):
        self._lock.lock()
        self._show_centers = show
        self._lock.unlock()

    def set_hidden_classes(self, classes: List[str]):
        self._lock.lock()
        self.hidden_classes = {c.strip().lower() for c in classes if c.strip()}
        self._lock.unlock()

    def set_info_panel_visibility(self, show: bool):
        self._lock.lock()
        self.show_info_panel = show
        self._lock.unlock()

    def set_info_panel_fields(self, fields: Dict[str, bool]):
        self._lock.lock()
        self.info_panel_fields.update(fields or {})
        self._lock.unlock()

    def set_model_path(self, model_path: str):
        model_path = (model_path or "").strip()
        if not model_path:
            return
        if not self.isRunning():
            self._reload_detector(model_path)
            return
        self._lock.lock()
        self._pending_model_path = model_path
        self._lock.unlock()
    
    def stop(self):
        """Stop processing."""
        self._running = False
        self.wait()
        try:
            if self.camera:
                self.camera.release()
        except Exception:
            pass

    def _reload_detector(self, model_path: str):
        """Recharge le détecteur YOLO avec un nouveau poids sans bloquer le thread."""
        model_path = model_path or self._model_path
        if model_path == self._model_path and self.detector is not None:
            return
        try:
            new_detector = Detector(model_path=model_path, conf=self._confidence)
        except Exception as e:
            self.error_occurred.emit(self.feed_id, f"Modèle indisponible ({model_path}): {e}")
            return
        # Remplacer l'ancien détecteur par le nouveau en douceur
        old_detector = self.detector
        self.detector = new_detector
        self._model_path = model_path
        if old_detector:
            try:
                old_detector.shutdown()
            except Exception:
                pass
        print(f"[VideoThread] Detector switched to {model_path}")