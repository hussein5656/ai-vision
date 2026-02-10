from typing import List, Dict, Any
from logic.events import Event, EventType
import time
import math

class AnomalyRule:
    """Détecte les vitesses anormales et les entrées dans les zones interdites."""

    def __init__(self, max_speed: float = 100.0, forbidden_zones: List[Dict[str, int]] = None):
        """
        max_speed : vitesse maximale en pixels/sec avant anomalie
        forbidden_zones : liste de rectangles à surveiller [{"x1":..,"y1":..,"x2":..,"y2":..,"zone_id":..}]
        """
        self.max_speed = max_speed
        self.forbidden_zones = forbidden_zones or []

        # mémoire interne : track_id -> {"last_pos":(x,y), "last_time":timestamp}
        self.memory: Dict[int, Dict[str, Any]] = {}

    def process(self, detections: List[Dict[str, Any]], global_memory: Dict[int, Dict[str, Any]]) -> List[Event]:
        events: List[Event] = []

        now = time.time()

        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]

            # vitesse
            if tid in self.memory:
                last_x, last_y = self.memory[tid]["last_pos"]
                last_time = self.memory[tid]["last_time"]
                dt = now - last_time
                if dt > 0:
                    # Calculer un module de vitesse simple en pixels/seconde
                    speed = math.sqrt((cx - last_x)**2 + (cy - last_y)**2) / dt
                    if speed > self.max_speed:
                        events.append(Event(
                            _event_type=EventType.ANOMALY,
                            track_id=tid,
                            details={"reason": f"speed {speed:.1f}px/sec"}
                        ))

            # zones interdites
            for zone in self.forbidden_zones:
                # Les zones sont décrites avec des bounding boxes simples pour limiter le coût CPU
                if zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]:
                    events.append(Event(
                        _event_type=EventType.ANOMALY,
                        track_id=tid,
                        zone_id=zone.get("zone_id"),
                        details={"reason": "forbidden_zone"}
                    ))

            # mettre à jour mémoire pour la prochaine frame
            self.memory[tid] = {"last_pos": (cx, cy), "last_time": now}

        return events
