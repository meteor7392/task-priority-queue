import heapq
import time
from threading import Lock
from typing import Any, Tuple, Iterator, Generator, List, Dict, Set, Optional
from contextlib import contextmanager

class PriorityQueueError(Exception):
    """Base exception for AgingPriorityQueue errors."""
    pass

class ItemNotFoundError(PriorityQueueError):
    """Raised when an item is not found in the queue."""
    pass

class QueueEmptyError(PriorityQueueError, IndexError):
    """Raised when performing an operation on an empty queue."""
    pass

class AgingPriorityQueue:
    """
    A Priority Queue where the priority of an item increases (numerically decreases)
    the longer it stays in the queue to prevent starvation.
    """
    def __init__(self, aging_rate: float = 0.1, min_priority: Optional[float] = None):
        """
        :param aging_rate: How much to decrease the priority value per second of waiting.
        :param min_priority: Optional lower bound for the effective priority. 
                             If set, priorities will not drop below this value.
        """
        self._queue = []
        self._lock = Lock()
        self.aging_rate = aging_rate
        self.min_priority = min_priority
        self._item_rates = {}
        self._items_set: Set[Any] = set()

    def set_aging_rate(self, new_rate: float):
        """
        Dynamically updates the aging rate for all items in the queue.
        """
        with self._lock:
            self.aging_rate = new_rate

    def set_min_priority(self, new_min: Optional[float]):
        """
        Updates the minimum allowed effective priority.
        """
        with self._lock:
            self.min_priority = new_min

    def set_item_aging_rate(self, item: Any, new_rate: float):
        """
        Sets a custom aging rate for a specific item, overriding the global rate.
        """
        with self._lock:
            if item not in self._items_set:
                raise ItemNotFoundError("item not in priority queue")
            self._item_rates[item] = new_rate

    def update_item_aging_rate(self, item: Any, new_rate: float):
        """
        Updates the custom aging rate for a specific item if it already has one.
        If the item is in the queue but has no custom rate, it will set one.
        """
        with self._lock:
            if item not in self._items_set:
                raise ItemNotFoundError("item not in priority queue")
            self._item_rates[item] = new_rate

    def remove_item_aging_rate(self, item: Any):
        """
        Removes the custom aging rate for a specific item, reverting it to the global rate.
        """
        with self._lock:
            if item not in self._items_set:
                raise ItemNotFoundError("item not in priority queue")
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

    def _find_item_index(self, item: Any) -> int:
        """Internal helper to find the index of an item in the queue. Raises ItemNotFoundError if not found."""
        for i, entry in enumerate(self._queue):
            if entry[2] == item:
                return i
        raise ItemNotFoundError("item not in priority queue")

    def _calculate_effective_priority(self, base_p: float, entry_time: float, item: Any, now: float) -> float:
        """Internal helper to calculate clamped effective priority."""
        rate = self._item_rates.get(item, self.aging_rate)
        effective = base_p - ((now - entry_time) * rate)
        if self.min_priority is not None:
            return max(effective, self.min_priority)
        return effective

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
            # The heap naturally orders by priority then entry_time (FIFO tie-break)
            heapq.heappush(self._queue, (priority, entry_time, item))
            self._items_set.add(item)

    def push_many(self, items: List[Tuple[float, Any]]):
        """
        Adds multiple items to the queue in a single lock acquisition.
        :param items: A list of tuples (priority, item).
        """
        with self._lock:
            for priority, item in items:
                if not isinstance(priority, (int, float)):
                    raise TypeError(f"Priority for item {item} must be a number")
                # Using time.time() inside the loop to ensure distinct entry times
                # for better FIFO stability, though they will be very close.
                heapq.heappush(self._queue, (priority, time.time(), item))
                self._items_set.add(item)

    def _find_best_entry(self, now: float) -> Tuple[int, Tuple[float, float, Any]]:
        """Internal helper to find the item with the highest effective priority."""
        # Optimization: If there are no per-item rates, the heap root is always the best 
        # because all items age at the same rate relative to their entry time.
        # effective_p = base_p - (now - entry_time) * global_rate
        # effective_p = (base_p + entry_time * global_rate) - now * global_rate
        # However, the current heap is sorted by (base_p, entry_time). 
        # Since we support per-item rates, we must scan unless _item_rates is empty.
        
        if not self._item_rates:
            # With uniform aging, the root of the min-heap (base_p, entry_time) 
            # is the highest priority candidate.
            return 0, self._queue[0]

        best_idx = -1
        best_priority = float('inf')
        
        for i, (base_p, entry_time, item) in enumerate(self._queue):
            current_priority = self._calculate_effective_priority(base_p, entry_time, item, now)
            if current_priority < best_priority:
                best_priority = current_priority
                best_idx = i
            elif current_priority == best_priority:
                # Stability: prefer older item if priorities are equal
                if best_idx != -1 and entry_time < self._queue[best_idx][1]:
                    best_idx = i

        return best_idx, self._queue[best_idx]

    def peek(self) -> Any:
        """
        Returns the item with the highest priority (lowest numerical value),
        accounting for aging, without removing it from the queue.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("peek from an empty priority queue")

            _, entry = self._find_best_entry(time.time())
            return entry[2]

    def get_top_priority(self) -> float:
        """
        Returns the effective priority value of the item that would be popped next.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("get_top_priority from an empty priority queue")

            now = time.time()
            idx, entry = self._find_best_entry(now)
            return self._calculate_effective_priority(entry[0], entry[1], entry[2], now)

    def get_top_item_details(self) -> Dict[str, Any]:
        """
        Returns the details (item, base_priority, entry_time) of the item that would be popped next.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("get_top_item_details from an empty priority queue")

            now = time.time()
            idx, entry = self._find_best_entry(now)
            base_p, entry_time, item = entry
            return {
                "item": item,
                "base_priority": base_p,
                "entry_time": entry_time,
                "effective_priority": self._calculate_effective_priority(base_p, entry_time, item, now)
            }

    def peek_bottom(self) -> Any:
        """
        Returns the item with the lowest priority (highest numerical value),
        accounting for aging, without removing it from the queue.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("peek_bottom from an empty priority queue")
            
            details = self.get_bottom_item_details()
            return details["item"]

    def get_bottom_priority(self) -> float:
        """
        Returns the effective priority value of the item with the lowest priority.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("get_bottom_priority from an empty priority queue")
            
            details = self.get_bottom_item_details()
            return details["effective_priority"]

    def get_bottom_item_details(self) -> Dict[str, Any]:
        """
        Returns the details (item, base_priority, entry_time) of the item that has the lowest
        effective priority (highest numerical value).
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("get_bottom_item_details from an empty priority queue")

            now = time.time()
            worst_entry = None
            worst_priority = float('-inf')
            
            for entry in self._queue:
                base_p, entry_time, item = entry
                current_priority = self._calculate_effective_priority(base_p, entry_time, item, now)
                if current_priority > worst_priority:
                    worst_priority = current_priority
                    worst_entry = entry

            base_p, entry_time, item = worst_entry
            return {
                "item": item,
                "base_priority": base_p,
                "entry_time": entry_time,
                "effective_priority": worst_priority
            }

    def pop(self) -> Any:
        """
        Removes and returns the item with the highest priority (lowest numerical value),
        accounting for aging.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("pop from an empty priority queue")

            best_idx, entry = self._find_best_entry(time.time())
            item = entry[2]
            
            if item in self._item_rates:
                del self._item_rates[item]
            self._items_set.remove(item)
            
            last_element = self._queue.pop()
            if best_idx < len(self._queue):
                self._queue[best_idx] = last_element
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
        Raises ItemNotFoundError if the item is not found.
        """
        with self._lock:
            idx = self._find_item_index(item)

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
        Raises ItemNotFoundError if the item is not found.
        """
        if not isinstance(new_priority, (int, float)):
            raise TypeError("Priority must be a number")

        with self._lock:
            idx = self._find_item_index(item)

            entry_time = self._queue[idx][1]
            self._queue[idx] = (new_priority, entry_time, item)
            
            heapq._siftdown(self._queue, 0, idx)
            heapq._siftup(self._queue, idx)

    def update_priorities_many(self, updates: List[Tuple[Any, float]]):
        """
        Updates the base priorities of multiple existing items while preserving their entry times.
        :param updates: A list of tuples (item, new_priority).
        Raises ItemNotFoundError if any item is not found.
        Raises TypeError if any priority is not a number.
        """
        with self._lock:
            # Validate all inputs first to ensure atomicity of the update
            for item, priority in updates:
                if not isinstance(priority, (int, float)):
                    raise TypeError(f"Priority for item {item} must be a number")
                if item not in self._items_set:
                    raise ItemNotFoundError(f"item {item} not in priority queue")

            for item, priority in updates:
                idx = self._find_item_index(item)
                entry_time = self._queue[idx][1]
                self._queue[idx] = (priority, entry_time, item)
                
                heapq._siftdown(self._queue, 0, idx)
                heapq._siftup(self._queue, idx)

    def adjust_priority(self, item: Any, delta: float):
        """
        Adjusts the base priority of an existing item by a given delta.
        Positive delta decreases priority (increases value), negative increases priority.
        Raises ItemNotFoundError if the item is not found.
        """
        if not isinstance(delta, (int, float)):
            raise TypeError("Delta must be a number")

        with self._lock:
            idx = self._find_item_index(item)

            base_p, entry_time, _ = self._queue[idx]
            self._queue[idx] = (base_p + delta, entry_time, item)
            
            heapq._siftdown(self._queue, 0, idx)
            heapq._siftup(self._queue, idx)

    def get_current_priority(self, item: Any) -> float:
        """
        Returns the current calculated priority of an item, accounting for aging.
        Raises ItemNotFoundError if the item is not found.
        """
        with self._lock:
            now = time.time()
            for base_p, entry_time, queue_item in self._queue:
                if queue_item == item:
                    return self._calculate_effective_priority(base_p, entry_time, queue_item, now)
            raise ItemNotFoundError("item not in priority queue")

    def get_base_priority(self, item: Any) -> float:
        """
        Returns the original base priority of an item without aging.
        Raises ItemNotFoundError if the item is not found.
        """
        with self._lock:
            for base_p, _, queue_item in self._queue:
                if queue_item == item:
                    return base_p
            raise ItemNotFoundError("item not in priority queue")

    def get_all_current_priorities(self) -> Dict[Any, float]:
        """
        Returns a dictionary mapping all items in the queue to their current
        calculated priorities.
        """
        with self._lock:
            now = time.time()
            priorities = {}
            for base_p, entry_time, item in self._queue:
                priorities[item] = self._calculate_effective_priority(base_p, entry_time, item, now)
            return priorities

    def get_priorities_for_items(self, items: List[Any]) -> Dict[Any, float]:
        """
        Returns a dictionary mapping the requested items to their current
        calculated priorities. Only includes items that are present in the queue.
        """
        with self._lock:
            now = time.time()
            requested_set = set(items)
            priorities = {}
            for base_p, entry_time, item in self._queue:
                if item in requested_set:
                    priorities[item] = self._calculate_effective_priority(base_p, entry_time, item, now)
            return priorities

    def get_priority_details(self, item: Any) -> Dict[str, Any]:
        """
        Returns the base priority and entry time of an item.
        Raises ItemNotFoundError if the item is not found.
        """
        with self._lock:
            for base_p, entry_time, queue_item in self._queue:
                if queue_item == item:
                    return {
                        "base_priority": base_p,
                        "entry_time": entry_time
                    }
            raise ItemNotFoundError("item not in priority queue")

    def get_sorted_tasks(self) -> List[Any]:
        """
        Returns a list of all items in the queue, sorted by their current
        effective priority (lowest value first).
        """
        with self._lock:
            now = time.time()
            tasks_with_priority = []
            for base_p, entry_time, item in self._queue:
                effective_priority = self._calculate_effective_priority(base_p, entry_time, item, now)
                tasks_with_priority.append((effective_priority, entry_time, item))
            
            # Sort by effective priority, then by entry_time for stability
            tasks_with_priority.sort()
            return [item for _, _, item in tasks_with_priority]

    def get_sorted_iterator(self) -> Iterator[Any]:
        """
        Returns an iterator that yields items in their current
        effective priority order.
        """
        return iter(self.get_sorted_tasks())

    def get_items_in_range(self, min_p: float, max_p: float) -> List[Any]:
        """
        Returns a list of items whose current effective priority falls within [min_p, max_p].
        """
        with self._lock:
            now = time.time()
            result = []
            for base_p, entry_time, item in self._queue:
                effective = self._calculate_effective_priority(base_p, entry_time, item, now)
                if min_p <= effective <= max_p:
                    result.append(item)
            return result

    def get_all_items(self) -> Set[Any]:
        """
        Returns a set of all items currently in the queue.
        """
        with self._lock:
            return set(self._items_set)

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
            try:
                idx = self._find_item_index(item)
                old_priority, entry_time, _ = self._queue[idx]
                self._queue[idx] = (old_priority - boost_amount, entry_time, item)
                heapq._siftdown(self._queue, 0, idx)
                heapq._siftup(self._queue, idx)
            except ItemNotFoundError:
                # If item is not in queue, the boost cannot be applied, but we let the context continue
                pass

        try:
            yield
        finally:
            with self._lock:
                try:
                    idx = self._find_item_index(item)
                    # We must retrieve the currently boosted priority to revert it
                    # Since priority_boost just subtracts, we add it back
                    current_priority, entry_time, _ = self._queue[idx]
                    self._queue[idx] = (current_priority + boost_amount, entry_time, item)
                    heapq._siftdown(self._queue, 0, idx)
                    heapq._siftup(self._queue, idx)
                except ItemNotFoundError:
                    pass

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

    def contains(self, item: Any) -> bool:
        """
        Returns True if the item is in the queue, False otherwise.
        """
        with self._lock:
            return item in self._items_set

    def __len__(self):
        return self.size()

    def __contains__(self, item: Any) -> bool:
        """
        Checks if an item is currently in the queue.
        """
        return self.contains(item)

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator over the items currently in the queue
        in the order they are stored internally.
        """
        with self._lock:
            return iter(self._queue_items())

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
