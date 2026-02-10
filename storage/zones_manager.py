"""Zones manager: save/load zones to JSON file."""

import copy
import json
import os
from typing import Dict, List, Any, Optional


class ZonesManager:
    """Manage persistence of zone configurations, per camera source."""

    ZONES_FILE = "zones.json"
    DEFAULT_KEYS = [
        'counting_lines',
        'forbidden_zones',
        'parking_zones',
        'loitering_zones',
        'restricted_areas'
    ]

    @staticmethod
    def save_zones(zones: Dict[str, List[Any]], camera_key: Optional[Any] = None) -> bool:
        """Save zones to JSON file (optionally scoped to a camera)."""
        try:
            data = ZonesManager._load_all()
            profile = ZonesManager._coerce_zones(zones)
            key = ZonesManager._normalize_camera_key(camera_key)
            if key is None:
                data['global'] = profile
            else:
                data.setdefault('cameras', {})[key] = profile
            ZonesManager._write_all(data)
            scope = key or 'global'
            print(f"[ZonesManager] Zones saved ({scope}) to {ZonesManager.ZONES_FILE}")
            return True
        except Exception as e:
            print(f"[ZonesManager] Error saving zones: {e}")
            return False

    @staticmethod
    def load_zones(camera_key: Optional[Any] = None) -> Dict[str, List[Any]]:
        """Load zones (global or per camera)."""
        data = ZonesManager._load_all()
        key = ZonesManager._normalize_camera_key(camera_key)
        if key and key in data.get('cameras', {}):
            zones = ZonesManager._coerce_zones(data['cameras'][key])
            print(f"[ZonesManager] Zones loaded for {key} from {ZonesManager.ZONES_FILE}")
            return zones
        default = ZonesManager._coerce_zones(data.get('global', {}))
        if key:
            print(f"[ZonesManager] No zones for {key}, using global/default profile")
        else:
            print(f"[ZonesManager] Zones loaded from {ZonesManager.ZONES_FILE}")
        return default

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_camera_key(camera_key: Optional[Any]) -> Optional[str]:
        if camera_key is None:
            return None
        try:
            return str(camera_key)
        except Exception:
            return None

    @staticmethod
    def _empty_zones() -> Dict[str, List[Any]]:
        return {key: [] for key in ZonesManager.DEFAULT_KEYS}

    @staticmethod
    def _coerce_zones(zones: Any) -> Dict[str, List[Any]]:
        result = ZonesManager._empty_zones()
        if isinstance(zones, dict):
            for key in result.keys():
                value = zones.get(key)
                if isinstance(value, list):
                    result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _default_store() -> Dict[str, Any]:
        return {
            'global': ZonesManager._empty_zones(),
            'cameras': {}
        }

    @staticmethod
    def _load_all() -> Dict[str, Any]:
        if not os.path.isfile(ZonesManager.ZONES_FILE):
            return ZonesManager._default_store()
        try:
            with open(ZonesManager.ZONES_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[ZonesManager] Error loading zones: {e}")
            return ZonesManager._default_store()

        # If file already structured with global/cameras
        if isinstance(raw, dict) and ('global' in raw or 'cameras' in raw):
            store = ZonesManager._default_store()
            store['global'] = ZonesManager._coerce_zones(raw.get('global', {}))
            if isinstance(raw.get('cameras'), dict):
                for key, value in raw['cameras'].items():
                    if key is not None:
                        store['cameras'][str(key)] = ZonesManager._coerce_zones(value)
            return store

        # Legacy format (single zone dict)
        if isinstance(raw, dict):
            store = ZonesManager._default_store()
            store['global'] = ZonesManager._coerce_zones(raw)
            return store

        return ZonesManager._default_store()

    @staticmethod
    def _write_all(data: Dict[str, Any]) -> None:
        with open(ZonesManager.ZONES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
