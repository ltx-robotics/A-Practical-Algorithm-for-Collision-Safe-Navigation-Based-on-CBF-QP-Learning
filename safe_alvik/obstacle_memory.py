

import math
from typing import List, Optional, Sequence

import numpy as np

from .config import MemoryConfig
from .geometry import world_to_body

QUADRANT_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


class ObstacleMemory:
    def __init__(self, cfg: MemoryConfig, endpoint_range_m: float):
        self.cfg = cfg
        self.endpoint_range_m = float(endpoint_range_m)
        # Rows are (x, y, last_seen_seconds, frames_seen).
        self._tracks: List[List[float]] = []

    def reset(self) -> None:
        self._tracks = []

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self._confirmed())

    def _confirmed(self) -> List[List[float]]:
        """Tracks seen in enough separate frames to be treated as real.

        A single false-low ToF sample would otherwise create a phantom obstacle
        that the QP then has to respect for a whole TTL. The board's own safety
        layer requires three consecutive samples for the same reason; requiring
        a few frames here decouples "how long do I remember a real surface"
        from "how easily do I hallucinate one".
        """
        needed = int(self.cfg.confirm_frames)
        if needed <= 1:
            return self._tracks
        return [t for t in self._tracks if t[3] >= needed]

    def points(self) -> np.ndarray:
        tracks = self._confirmed()
        if not tracks:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([[t[0], t[1]] for t in tracks], dtype=np.float64)

    def ages(self, now_s: float) -> np.ndarray:
        tracks = self._confirmed()
        if not tracks:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray([now_s - t[2] for t in tracks], dtype=np.float64)

    # ----------------------------------------------------------------- update
    def update(self, endpoints: Optional[np.ndarray], now_s: float) -> None:
        self._expire(now_s)
        if endpoints is None or len(endpoints) == 0:
            return

        alpha = float(self.cfg.update_alpha)
        for point in np.asarray(endpoints, dtype=np.float64).reshape(-1, 2):
            index = self._nearest(point)
            if index is None:
                self._tracks.append([float(point[0]), float(point[1]), float(now_s), 1.0])
            else:
                track = self._tracks[index]
                track[0] = (1.0 - alpha) * track[0] + alpha * float(point[0])
                track[1] = (1.0 - alpha) * track[1] + alpha * float(point[1])
                # Several zones can hit one surface in the same frame, so
                # count frames rather than endpoints.
                if float(now_s) > track[2]:
                    track[3] += 1.0
                track[2] = float(now_s)

        if len(self._tracks) > self.cfg.max_tracks:
            # Keep confirmed surfaces first, then the most recently observed,
            # so an unconfirmed phantom cannot evict a real obstacle.
            needed = int(self.cfg.confirm_frames)
            self._tracks.sort(key=lambda t: (t[3] >= needed, t[2]), reverse=True)
            self._tracks = self._tracks[: self.cfg.max_tracks]

    def _expire(self, now_s: float) -> None:
        ttl = float(self.cfg.ttl_s)
        self._tracks = [t for t in self._tracks if now_s - t[2] <= ttl]

    def _nearest(self, point: np.ndarray) -> Optional[int]:
        best_index: Optional[int] = None
        best_distance = self.cfg.cluster_radius_m
        for index, track in enumerate(self._tracks):
            distance = math.hypot(track[0] - point[0], track[1] - point[1])
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    # --------------------------------------------------------------- features
    def quadrant_features(self, odom_pose: Sequence[float]) -> np.ndarray:
        """Closest remembered surface per body quadrant, as a proximity in [0, 1].

        Order matches the observation layout:
        (front_left, front_right, rear_left, rear_right).
        """
        features = np.zeros(4, dtype=np.float64)
        tracks = self._confirmed()
        if not tracks:
            return features
        for track in tracks:
            body = world_to_body(odom_pose, (track[0], track[1]))
            distance = float(math.hypot(body[0], body[1]))
            proximity = 1.0 - distance / self.endpoint_range_m
            if proximity <= 0.0:
                continue
            if body[0] >= 0.0:
                slot = 0 if body[1] >= 0.0 else 1
            else:
                slot = 2 if body[1] >= 0.0 else 3
            if proximity > features[slot]:
                features[slot] = proximity
        return features
