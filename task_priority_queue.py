import heapq
import time
from threading import Lock
from typing import Any, Tuple

class AgingPriorityQueue:
    """
    A Priority Queue where the priority of an item increases (numerically decreases)
    the longer it stays in the queue to prevent starvation.
    """
    def __init__(self, aging_rate: float = 0.1):
        """
        :param aging_rate: How much to decrease the priority value per second of waiting.
        """
        self._queue = []
        self._lock = Lock()
        self.aging_rate = aging_rate

    def push(self, priority: float, item: Any):
        """
        Adds an item to the queue.
        Priority is stored as (current_priority, entry_time, item).
        """
        with self._lock:
            # We store entry time to calculate age during pop
            entry_time = time.time()
            heapq.heappush(self._queue, (priority, entry_time, item))

    def peek(self) -> Any:
        """
        Returns the item with the highest priority (lowest numerical value),
        accounting for aging, without removing it from the queue.
        """
        with self._lock:
            if not self._queue:
                raise IndexError("peek from an empty priority queue")

            now = time.time()
            best_idx = -1
            best_priority = float('inf')

            for i, (orig_priority, entry_time, item) in enumerate(self._queue):
                age = now - entry_time
                current_priority = orig_priority - (age * self.aging_rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_idx = i

            return self._queue[best_idx][2]

    def pop(self) -> Any:
        """
        Removes and returns the item with the highest priority (lowest numerical value),
        accounting for aging.
        """
        with self._lock:
            if not self._queue:
                raise IndexError("pop from an empty priority queue")

            # To accurately pop the highest priority with aging,
            # we must evaluate all items because a very old low-priority 
            # task might now be the highest priority.
            # For performance in large queues, this could be optimized
            # with a bucket-based approach, but for this utility, 
            # we recalculate the best candidate.
            
            now = time.time()
            best_idx = -1
            best_priority = float('inf')

            for i, (orig_priority, entry_time, item) in enumerate(self._queue):
                age = now - entry_time
                current_priority = orig_priority - (age * self.aging_rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_idx = i

            # Remove the best element and maintain heap property
            item = self._queue[best_idx][2]
            self._queue[best_idx] = self._queue[-1]
            self._queue.pop()
            if best_idx < len(self._queue):
                heapq._siftdown(self._queue, 0, best_idx)
                heapq._siftup(self._queue, best_idx)
            
            return item

    def __len__(self):
        with self._lock:
            return len(self._queue)