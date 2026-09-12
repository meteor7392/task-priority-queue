import heapq
import time
from threading import Lock
from typing import Any, Tuple, Iterator, Generator
from contextlib import contextmanager

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
            best_item = None
            best_priority = float('inf')

            for _, entry_time, item in self._queue:
                age = now - entry_time
                current_priority = _ - (age * self.aging_rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_item = item

            # Re-scanning for the item in the heap to ensure we return the correct one
            # if priorities are identical. 
            return best_item

    def pop(self) -> Any:
        """
        Removes and returns the item with the highest priority (lowest numerical value),
        accounting for aging.
        """
        with self._lock:
            if not self._queue:
                raise IndexError("pop from an empty priority queue")

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

    def remove(self, item: Any):
        """
        Removes a specific item from the queue if it exists.
        Raises ValueError if the item is not found.
        """
        with self._lock:
            idx = -1
            for i, entry in enumerate(self._queue):
                if entry[2] == item:
                    idx = i
                    break
            
            if idx == -1:
                raise ValueError("item not in priority queue")

            self._queue[idx] = self._queue[-1]
            self._queue.pop()
            if idx < len(self._queue):
                heapq._siftdown(self._queue, 0, idx)
                heapq._siftup(self._queue, idx)

    def update_priority(self, item: Any, new_priority: float):
        """
        Updates the base priority of an existing item while preserving its entry time.
        Raises ValueError if the item is not found.
        """
        with self._lock:
            idx = -1
            for i, entry in enumerate(self._queue):
                if entry[2] == item:
                    idx = i
                    break
            
            if idx == -1:
                raise ValueError("item not in priority queue")

            # Preserve the original entry time to maintain the aging progress
            entry_time = self._queue[idx][1]
            self._queue[idx] = (new_priority, entry_time, item)
            
            # Since we changed the priority, we must restore the heap property
            heapq._siftdown(self._queue, 0, idx)
            heapq._siftup(self._queue, idx)

    def get_current_priority(self, item: Any) -> float:
        """
        Returns the current calculated priority of an item, accounting for aging.
        Raises ValueError if the item is not found.
        """
        with self._lock:
            now = time.time()
            for orig_priority, entry_time, queue_item in self._queue:
                if queue_item == item:
                    age = now - entry_time
                    return orig_priority - (age * self.aging_rate)
            raise ValueError("item not in priority queue")

    def clear(self):
        """
        Removes all items from the queue.
        """
        with self._lock:
            self._queue.clear()

    @contextmanager
    def priority_boost(self, item: Any, boost_amount: float) -> Generator[None, None, None]:
        """
        Temporarily decreases the priority value (increases priority) of an item
        for the duration of the context block.
        """
        with self._lock:
            idx = -1
            for i, entry in enumerate(self._queue):
                if entry[2] == item:
                    idx = i
                    break
            
            if idx == -1:
                raise ValueError("item not in priority queue")
            
            old_priority, entry_time, _ = self._queue[idx]
            self._queue[idx] = (old_priority - boost_amount, entry_time, item)
            heapq._siftdown(self._queue, 0, idx)
            heapq._siftup(self._queue, idx)

        try:
            yield
        finally:
            with self._lock:
                # Restore the original priority
                idx = -1
                for i, entry in enumerate(self._queue):
                    if entry[2] == item:
                        idx = i
                        break
                if idx != -1:
                    _, entry_time, _ = self._queue[idx]
                    self._queue[idx] = (old_priority, entry_time, item)
                    heapq._siftdown(self._queue, 0, idx)
                    heapq._siftup(self._queue, idx)

    def __len__(self):
        with self._lock:
            return len(self._queue)

    def __contains__(self, item: Any) -> bool:
        """
        Checks if an item is currently in the queue.
        """
        with self._lock:
            return any(entry[2] == item for entry in self._queue)

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator over the items currently in the queue.
        Note: The order of iteration is based on the underlying heap structure,
        not the current aged priority order.
        """
        with self._lock:
            # Return a snapshot of items to ensure thread-safety during iteration
            return iter([entry[2] for entry in self._queue])