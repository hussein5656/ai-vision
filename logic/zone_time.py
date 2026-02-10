from typing import List, Dict, Any, Optional, Set
from logic.events import Event, EventType

class ZoneTimeRule:
    def __init__(self, zone_id: str, x1: int, y1: int, x2: int, y2: int,
                 time_threshold: float, allowed_classes: Optional[Set[int]] = None):
        """
        zone_id : identifiant de la zone
        x1, y1, x2, y2 : coordonnées du rectangle (haut-gauche, bas-droite)
        time_threshold : temps en secondes avant déclenchement
        """
        self.zone_id = zone_id
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.time_threshold = time_threshold
        self.allowed_classes = allowed_classes

        # mémoire interne : track_id -> timestamp d'entrée
        self.enter_times: Dict[int, float] = {}

    def process(self, detections: List[Dict[str, Any]], memory: Dict[int, Dict[str, Any]]) -> List[Event]:
        import time
        events: List[Event] = []
        
        # Debug: log zone boundaries on first call
        if not hasattr(self, '_logged_bounds'):
            print(f"[ZoneTimeRule] Zone '{self.zone_id}' bounds: x({self.x1}-{self.x2}), y({self.y1}-{self.y2}), threshold={self.time_threshold}s")
            self._logged_bounds = True

        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]
            cls_id = det.get("class")

            if self.allowed_classes is not None and cls_id not in self.allowed_classes:
                continue

            in_zone = self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2
            
            # Debug: log first few detections near zone
            if not hasattr(self, '_logged_dets'):
                self._logged_dets = 0
            if self._logged_dets < 5 and (in_zone or abs(cx - (self.x1 + self.x2)//2) < 200):
                print(f"[ZoneTimeRule] Det ID={tid} center=({cx},{cy}) in_zone={in_zone}")
                self._logged_dets += 1

            # nouvelle entrée
            if in_zone and tid not in self.enter_times:
                self.enter_times[tid] = time.time()
                print(f"[ZoneTimeRule] ENTRY zone='{self.zone_id}' tid={tid}")
                events.append(Event(
                    _event_type=EventType.ZONE_ENTER,
                    track_id=tid,
                    zone_id=self.zone_id
                ))

            # sortie
            elif not in_zone and tid in self.enter_times:
                del self.enter_times[tid]
                events.append(Event(
                    _event_type=EventType.ZONE_EXIT,
                    track_id=tid,
                    zone_id=self.zone_id
                ))

            # vérifier temps dépassé
            elif in_zone and tid in self.enter_times:
                elapsed = time.time() - self.enter_times[tid]
                if elapsed >= self.time_threshold:
                    # On déclenche et on remet le timer à zéro pour éviter de spammer
                    print(f"[ZoneTimeRule] PARKING ALERT zone='{self.zone_id}' tid={tid} elapsed={elapsed:.1f}s")
                    events.append(Event(
                        _event_type=EventType.ZONE_TIME_EXCEEDED,
                        track_id=tid,
                        zone_id=self.zone_id,
                        details={"elapsed_time": elapsed}
                    ))
                    # reset timer to avoid duplicate alerts until threshold met again
                    self.enter_times[tid] = time.time()

        return events

    def init_with_detections(self, detections: List[Dict[str, Any]]):
        """Initialize enter_times with objects already in zone.
        
        Call this when zone is first created to count objects that are
        already present.
        """
        import time
        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]
            cls_id = det.get("class")
            if self.allowed_classes is not None and cls_id not in self.allowed_classes:
                continue
            in_zone = self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2
            if in_zone and tid not in self.enter_times:
                self.enter_times[tid] = time.time()
