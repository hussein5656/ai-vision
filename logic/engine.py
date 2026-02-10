from typing import List, Dict, Any

from logic.events import Event

class LogicEngine:
    """Orchestre l'exécution séquentielle de toutes les règles métiers."""

    def __init__(self, rules: list):
        """
        rules : liste d'objets logiques
        ex: [LineCrossingRule(), ZoneTimeRule(), ...]
        """
        self.rules = rules

        # Mémoire globale par track_id
        self.memory: Dict[int, Dict[str, Any]] = {}
    def _update_memory(self, detections: List[dict]):
        for det in detections:
            track_id = det["track_id"]

            if track_id not in self.memory:
                self.memory[track_id] = {}

            self.memory[track_id]["last_detection"] = det

    def process_frame(self, detections: List[dict]) -> List[Event]:
        """
        detections : liste de détections IA pour UNE frame
        retourne : liste d'événements logiques
        """
        events: List[Event] = []

        # 1. Mettre à jour la mémoire
        self._update_memory(detections)

        # 2. Appliquer chaque règle logique dans l'ordre déclaré
        for rule in self.rules:
            rule_events = rule.process(detections, self.memory)
            if rule_events:
                events.extend(rule_events)

        return events
