import heapq
import time
from threading import Lock
from typing import Any, Tuple, Iterator, Generator, List
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

    def set_aging_rate(self, new_rate: float):
        """
        Dynamically updates the aging rate for all items in the queue.
        """
        with self._lock:
            self.aging_rate = new_rate

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

            # Since aging is uniform across all elements, the item that is currently
            # the highest priority is either the one with the lowest base priority
            # or the one that has been waiting the longest.
            # To be accurate with aging, we must find the minimum of (base_p - rate * age).
            for base_p, entry_time, item in self._queue:
                current_priority = base_p - ((now - entry_time) * self.aging_rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_item = item

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

            for i, (base_p, entry_time, item) in enumerate(self._queue):
                current_priority = base_p - ((now - entry_time) * self.aging_rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_idx = i

            # Remove the best element
            item = self._queue[best_idx][2]
            
            # To maintain heap property after removing an arbitrary index:
            # 1. Swap with the last element
            # 2. Pop the last element
            # 3. Restore heap property for the swapped element
            last_element = self._queue.pop()
            if best_idx < len(self._queue):
                self._queue[best_idx] = last_element
                # Sift down and up to reposition the element
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

            last_element = self._queue.pop()
            if idx < len(self._queue):
                self._queue[idx] = last_element
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

            entry_time = self._queue[idx][1]
            self._queue[idx] = (new_priority, entry_time, item)
            
            heapq._siftdown(self._queue, 0, idx)
            heapq._siftup(self._queue, idx)

    def get_current_priority(self, item: Any) -> float:
        """
        Returns the current calculated priority of an item, accounting for aging.
        Raises ValueError if the item is not found.
        """
        with self._lock:
            now = time.time()
            for base_p, entry_time, queue_item in self._queue:
                if queue_item == item:
                    return base_p - ((now - entry_time) * self.aging_rate)
            raise ValueError("item not in priority queue")

    def get_sorted_tasks(self) -> List[Any]:
        """
        Returns a list of all items in the queue, sorted by their current
        effective priority (lowest value first).
        """
        with self._lock:
            now = time.time()
            tasks_with_priority = []
            for base_p, entry_time, item in self._queue:
                effective_priority = base_p - ((now - entry_time) * self.aging_rate)
                tasks_with_priority.append((effective_priority, item))
            
            tasks_with_priority.sort(key=lambda x: x[0])
            return [item for _, item in tasks_with_priority]

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

    def is_empty(self) -> bool:
        """
        Returns True if the queue is empty, False otherwise.
        """
        with self._lock:
            return len(self._queue) == 0

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
        """
        with self._lock:
            return iter([entry[2] for entry in self._queue])

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the queue.
        """
        with self._lock:
            return f"AgingPriorityQueue(aging_rate={self.aging_rate}, size={len(self._queue)})"

    def __str__(self) -> str:
        """
        Returns a human-readable string of the current sorted tasks.
        """
        tasks = self.get_sorted_tasks()
        return f"AgingPriorityQueue(tasks={tasks})"
