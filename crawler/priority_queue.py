"""Priority queue for crawl frontier management."""

from __future__ import annotations

import heapq
from crawler.models import CrawlAction


class CrawlFrontier:
    """Max-priority heap queue of CrawlAction items."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, CrawlAction]] = []
        self._counter = 0

    def push(self, action: CrawlAction) -> None:
        """Add an action to the frontier."""
        heapq.heappush(self._heap, (-action.priority_score, self._counter, action))
        self._counter += 1

    def pop(self) -> CrawlAction | None:
        """Pop the highest-priority action, or None if empty."""
        if not self._heap:
            return None
        _, _, action = heapq.heappop(self._heap)
        return action

    def is_empty(self) -> bool:
        """Return True when the frontier has no pending actions."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
