from enum import Enum
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any, Union


class EventType(Enum):
    LINE_IN = "line_in"
    LINE_OUT = "line_out"

    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    ZONE_TIME_EXCEEDED = "zone_time_exceeded"

    LOITERING = "loitering"
    ANOMALY = "anomaly"


@dataclass
class Event:
    # Accepte soit EventType soit une simple chaîne pour rester flexible entre modules
    _event_type: Union[EventType, str]
    track_id: int
    timestamp: float = field(default_factory=time.time)

    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    line_id: Optional[str] = None

    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Normalized event type as string (e.g. 'line_in')."""
        if isinstance(self._event_type, EventType):
            return self._event_type.value
        return str(self._event_type)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "line_id": self.line_id,
            "details": self.details,
        }
