import heapq
import time
from threading import Lock
from typing import Any, Tuple, Iterator, Generator, List, Dict, Set
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
        self._item_rates = {}
        self._items_set: Set[Any] = set()

    def set_aging_rate(self, new_rate: float):
        """
        Dynamically updates the aging rate for all items in the queue.
        """
        with self._lock:
            self.aging_rate = new_rate

    def set_item_aging_rate(self, item: Any, new_rate: float):
        """
        Sets a custom aging rate for a specific item, overriding the global rate.
        """
        with self._lock:
            if item not in self._items_set:
                raise ValueError("item not in priority queue")
            self._item_rates[item] = new_rate

    def remove_item_aging_rate(self, item: Any):
        """
        Removes the custom aging rate for a specific item, reverting it to the global rate.
        """
        with self._lock:
            if item not in self._items_set:
                raise ValueError("item not in priority queue")
            if item in self._item_rates:
                del self._item_rates[item]

    def reset_all_item_aging_rates(self):
        """
        Removes all per-item aging rate overrides, causing all items
        to use the current global aging rate.
        """
        with self._lock:
            self._item_rates.clear()

    def _queue_items(self) -> List[Any]:
        """Internal helper to get all items in the queue."""
        return [entry[2] for entry in self._queue]

    def push(self, priority: float, item: Any):
        """
        Adds an item to the queue.
        Priority is stored as (current_priority, entry_time, item).
        """
        if not isinstance(priority, (int, float)):
            raise TypeError("Priority must be a number")
            
        with self._lock:
            # We store entry time to calculate age during pop
            entry_time = time.time()
            heapq.heappush(self._queue, (priority, entry_time, item))
            self._items_set.add(item)

    def push_many(self, items: List[Tuple[float, Any]]):
        """
        Adds multiple items to the queue in a single lock acquisition.
        :param items: A list of tuples (priority, item).
        """
        with self._lock:
            now = time.time()
            for priority, item in items:
                if not isinstance(priority, (int, float)):
                    raise TypeError(f"Priority for item {item} must be a number")
                heapq.heappush(self._queue, (priority, now, item))
                self._items_set.add(item)

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

            # Since aging can be per-item, we must check all elements
            for base_p, entry_time, item in self._queue:
                rate = self._item_rates.get(item, self.aging_rate)
                current_priority = base_p - ((now - entry_time) * rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_item = item

            return best_item

    def get_top_priority(self) -> float:
        """
        Returns the effective priority value of the item that would be popped next.
        """
        with self._lock:
            if not self._queue:
                raise IndexError("get_top_priority from an empty priority queue")

            now = time.time()
            best_priority = float('inf')

            for base_p, entry_time, item in self._queue:
                rate = self._item_rates.get(item, self.aging_rate)
                current_priority = base_p - ((now - entry_time) * rate)
                if current_priority < best_priority:
                    best_priority = current_priority

            return best_priority

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
                rate = self._item_rates.get(item, self.aging_rate)
                current_priority = base_p - ((now - entry_time) * rate)
                if current_priority < best_priority:
                    best_priority = current_priority
                    best_idx = i

            # Remove the best element
            item = self._queue[best_idx][2]
            
            # Cleanup item rate and membership
            if item in self._item_rates:
                del self._item_rates[item]
            self._items_set.remove(item)
            
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

    def pop_all(self) -> Generator[Any, None, None]:
        """
        Yields items from the queue one by one in priority order
        until the queue is empty.
        """
        while not self.is_empty():
            yield self.pop()

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

            if item in self._item_rates:
                del self._item_rates[item]
            self._items_set.remove(item)

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
        if not isinstance(new_priority, (int, float)):
            raise TypeError("Priority must be a number")

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
                    rate = self._item_rates.get(queue_item, self.aging_rate)
                    return base_p - ((now - entry_time) * rate)
            raise ValueError("item not in priority queue")

    def get_all_current_priorities(self) -> Dict[Any, float]:
        """
        Returns a dictionary mapping all items in the queue to their current
        calculated priorities.
        """
        with self._lock:
            now = time.time()
            priorities = {}
            for base_p, entry_time, item in self._queue:
                rate = self._item_rates.get(item, self.aging_rate)
                priorities[item] = base_p - ((now - entry_time) * rate)
            return priorities

    def get_priority_details(self, item: Any) -> Dict[str, Any]:
        """
        Returns the base priority and entry time of an item.
        Raises ValueError if the item is not found.
        """
        with self._lock:
            for base_p, entry_time, queue_item in self._queue:
                if queue_item == item:
                    return {
                        "base_priority": base_p,
                        "entry_time": entry_time
                    }
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
                rate = self._item_rates.get(item, self.aging_rate)
                effective_priority = base_p - ((now - entry_time) * rate)
                tasks_with_priority.append((effective_priority, item))
            
            tasks_with_priority.sort(key=lambda x: x[0])
            return [item for _, item in tasks_with_priority]

    def clear(self):
        """
        Removes all items from the queue.
        """
        with self._lock:
            self._queue.clear()
            self._item_rates.clear()
            self._items_set.clear()

    @contextmanager
    def priority_boost(self, item: Any, boost_amount: float) -> Generator[None, None, None]:
        """
        Temporarily decreases the priority value (increases priority) of an item
        for the duration of the context block.
        """
        if not isinstance(boost_amount, (int, float)):
            raise TypeError("Boost amount must be a number")

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

    def size(self) -> int:
        """
        Returns the current number of items in the queue.
        """
        with self._lock:
            return len(self._queue)

    def __len__(self):
        return self.size()

    def __contains__(self, item: Any) -> bool:
        """
        Checks if an item is currently in the queue.
        """
        with self._lock:
            return item in self._items_set

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
