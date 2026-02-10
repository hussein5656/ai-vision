"""Lightweight tracker to maintain stable `track_id` across frames.

This is a minimal proximity-based tracker: it matches detections to
existing tracks by nearest-center within a threshold. If no match is
found, it creates a new `track_id`.
"""

import math


class Tracker:
    def __init__(self, max_distance: int = 60):
        self.max_distance = max_distance
        self.next_id = 1
        # track_id -> last_center (x,y)
        self.tracks = {}

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, detections):
        updated = []
        used_ids = set()

        for det in detections:
            tid = det.get('track_id')
            center = tuple(int(c) for c in det.get('center', (0,0)))

            if tid is not None:
                # ensure track exists
                self.tracks[tid] = center
                updated.append(det)
                used_ids.add(tid)
                continue

            # try to match to existing tracks by proximity
            best_id = None
            best_dist = None
            for tr_id, last_center in self.tracks.items():
                if tr_id in used_ids:
                    continue
                d = self._dist(center, last_center)
                if d <= self.max_distance and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_id = tr_id

            if best_id is not None:
                det['track_id'] = best_id
                self.tracks[best_id] = center
                used_ids.add(best_id)
                updated.append(det)
            else:
                new_id = self.next_id
                self.next_id += 1
                det['track_id'] = new_id
                self.tracks[new_id] = center
                used_ids.add(new_id)
                updated.append(det)

        # Optionally, prune tracks that weren't seen this frame (simple)
        # Keep only tracks in used_ids
        self.tracks = {tid: self.tracks[tid] for tid in used_ids if tid in self.tracks}

        return updated
