"""In-process priority queue for clinician review."""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewItem:
    frame_id: int
    uncertainty: float


class ClinicianReviewQueue:
    """Highest uncertainty is dispatched first."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int]] = []
        self._sequence = 0

    def push(self, frame_id: int, uncertainty: float) -> None:
        self._sequence += 1
        heapq.heappush(self._heap, (-float(uncertainty), self._sequence, int(frame_id)))

    def pop(self) -> ReviewItem:
        if not self._heap:
            raise IndexError("clinician review queue is empty")
        negative_score, _sequence, frame_id = heapq.heappop(self._heap)
        return ReviewItem(frame_id=frame_id, uncertainty=-negative_score)

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def depth(self) -> int:
        return len(self._heap)
