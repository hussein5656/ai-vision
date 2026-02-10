from typing import List, Dict, Any, Optional
from logic.events import Event, EventType

class LineCrossingRule:
    def __init__(self, line_y: int = 300, direction: str = "horizontal",
                 x_start: int = 0, x_end: Optional[int] = None,
                 y_start: int = 0, y_end: Optional[int] = None):
        """
        line_y : position de la ligne (axe y pour horizontal, x pour vertical)
        direction : "horizontal" ou "vertical"
        x_start/x_end : plage horizontale couverte par la ligne (optionnel)
        y_start/y_end : plage verticale couverte pour les lignes verticales
        """
        self.line_y = line_y
        self.direction = direction
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

        # mémoire interne pour savoir qui a déjà franchi
        self.last_positions: Dict[int, int] = {}  # track_id -> dernier coord connu sur l'axe surveillé
        self.counted_in: set = set()
        self.counted_out: set = set()

    def process(self, detections: List[Dict[str, Any]], memory: Dict[int, Dict[str, Any]]) -> List[Event]:
        events: List[Event] = []
        
        # Debug: log line position on first call
        if not hasattr(self, '_logged_line'):
            print(f"[LineCrossingRule] Line at y={self.line_y}, direction={self.direction}")
            self._logged_line = True

        for det in detections:
            tid = det["track_id"]

            # récupérer la coordonnée à surveiller (y pour horizontal)
            if self.direction == "horizontal":
                coord = det["center"][1]
                axis_value = det["center"][0]
                # Filtrer les objets qui sont en dehors de la portion utile de la ligne
                if self.x_end is not None and axis_value > self.x_end:
                    continue
                if axis_value < self.x_start:
                    continue
            else:
                coord = det["center"][0]
                axis_value = det["center"][1]
                if self.y_end is not None and axis_value > self.y_end:
                    continue
                if axis_value < self.y_start:
                    continue

            prev = self.last_positions.get(tid)
            self.last_positions[tid] = coord
            
            # Debug: log first few detections
            if not hasattr(self, '_logged_dets'):
                self._logged_dets = 0
            if self._logged_dets < 5:
                print(f"[LineCrossingRule] Det ID={tid} coord={coord} prev={prev}")
                self._logged_dets += 1

            if prev is None:
                continue

            # haut -> bas = IN
            crossed_down = prev < self.line_y <= coord
            # bas -> haut = OUT
            crossed_up = prev > self.line_y >= coord
            
            if crossed_down or crossed_up:
                print(f"[LineCrossingRule] CROSS tid={tid} prev={prev} curr={coord} down={crossed_down} up={crossed_up}")

            if crossed_down and tid not in self.counted_in:
                self.counted_in.add(tid)
                print(f"[LineCrossingRule] ENTRY tid={tid}")
                events.append(Event(
                    _event_type=EventType.LINE_IN,
                    track_id=tid,
                    line_id="main_line",
                    details={"direction": "up_to_down"}
                ))

            if crossed_up and tid not in self.counted_out:
                self.counted_out.add(tid)
                print(f"[LineCrossingRule] EXIT tid={tid}")
                events.append(Event(
                    _event_type=EventType.LINE_OUT,
                    track_id=tid,
                    line_id="main_line",
                    details={"direction": "down_to_up"}
                ))

        return events
