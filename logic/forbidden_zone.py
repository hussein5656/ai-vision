"""Rule that triggers when a detection enters a forbidden polygon."""

from typing import List, Dict, Any
import numpy as np
import cv2

from logic.events import Event, EventType


class ForbiddenZoneRule:
    """Déclenche un événement ANOMALY dès qu'un objet entre dans le polygone interdit."""

    def __init__(self, zone_id: str, polygon_points: List[List[int]]):
        if polygon_points is None or len(polygon_points) < 3:
            raise ValueError("ForbiddenZoneRule requires at least 3 points")
        self.zone_id = zone_id
        self.polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        self.present_ids = set()

    def process(self, detections: List[Dict[str, Any]], memory: Dict[int, Dict[str, Any]]) -> List[Event]:
        events: List[Event] = []

        current_ids = set()
        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]
            # Utilisation de pointPolygonTest pour garder une tolérance précise sur les bords
            inside = cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False) >= 0
            if inside:
                current_ids.add(tid)
                if tid not in self.present_ids:
                    self.present_ids.add(tid)
                    events.append(Event(
                        _event_type=EventType.ANOMALY,
                        track_id=tid,
                        zone_id=self.zone_id,
                        details={"reason": "forbidden_zone"}
                    ))
            else:
                if tid in self.present_ids:
                    self.present_ids.remove(tid)

        # Remove IDs no longer seen
        stale_ids = self.present_ids - current_ids
        for tid in stale_ids:
            self.present_ids.remove(tid)

        return events
