from typing import List, Dict, Any
from logic.events import Event, EventType
import time

class LoiteringRule:
    def __init__(self, zone_id: str, x1: int, y1: int, x2: int, y2: int, time_threshold: float, movement_threshold: int = 10):
        """
        zone_id : identifiant de la zone
        x1, y1, x2, y2 : coordonnées du rectangle
        time_threshold : temps minimum avant déclenchement
        movement_threshold : distance minimale pour considérer que la personne bouge
        """
        self.zone_id = zone_id
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.time_threshold = time_threshold
        self.movement_threshold = movement_threshold

        # mémoire interne : track_id -> {"entry_time": timestamp, "last_pos": (x, y)}
        self.memory: Dict[int, Dict[str, Any]] = {}

    def process(self, detections: List[Dict[str, Any]], global_memory: Dict[int, Dict[str, Any]]) -> List[Event]:
        events: List[Event] = []

        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]

            in_zone = self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

            if not in_zone:
                # sort de la zone : supprimer de la mémoire
                if tid in self.memory:
                    del self.memory[tid]
                continue

            # première fois dans la zone
            if tid not in self.memory:
                self.memory[tid] = {"entry_time": time.time(), "last_pos": (cx, cy)}
                continue

            # vérifier distance parcourue
            last_x, last_y = self.memory[tid]["last_pos"]
            distance = ((cx - last_x)**2 + (cy - last_y)**2)**0.5

            elapsed = time.time() - self.memory[tid]["entry_time"]

            if distance <= self.movement_threshold and elapsed >= self.time_threshold:
                # Ici on considère que l'objet stagne et dépasse la durée maximale
                # déclenchement de l'événement
                events.append(Event(
                    _event_type=EventType.LOITERING,
                    track_id=tid,
                    zone_id=self.zone_id,
                    details={"elapsed_time": elapsed, "movement": distance}
                ))
                # reset pour éviter plusieurs events
                self.memory[tid]["entry_time"] = time.time()

            # mettre à jour position
            self.memory[tid]["last_pos"] = (cx, cy)

        return events
