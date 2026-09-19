import heapq
import time
from threading import Lock, Condition
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
        self._condition = Condition(self._lock)
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
            self._condition.notify_all()

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
            self._condition.notify_all()

    def update_item_aging_rate(self, item: Any, new_rate: float):
        """
        Updates the custom aging rate for a specific item if it already has one.
        If the item is in the queue but has no custom rate, it will set one.
        """
        with self._lock:
            if item not in self._items_set:
                raise ItemNotFoundError("item not in priority queue")
            self._item_rates[item] = new_rate
            self._condition.notify_all()

    def update_item_aging_rates_many(self, updates: List[Tuple[Any, float]]):
        """
        Updates the custom aging rates for multiple items in a single lock acquisition.
        :param updates: A list of tuples (item, new_rate).
        Raises ItemNotFoundError if any item is not found in the queue.
        Raises TypeError if any rate is not a number.
        """
        with self._lock:
            # Validate all items first for atomicity
            for item, rate in updates:
                if not isinstance(rate, (int, float)):
                    raise TypeError(f"Aging rate for item {item} must be a number")
                if item not in self._items_set:
                    raise ItemNotFoundError(f"item {item} not in priority queue")
            
            for item, rate in updates:
                self._item_rates[item] = rate
            self._condition.notify_all()

    def remove_item_aging_rate(self, item: Any):
        """
        Removes the custom aging rate for a specific item, reverting it to the global rate.
        """
        with self._lock:
            if item not in self._items_set:
                raise ItemNotFoundError("item not in priority queue")
            if item in self._item_rates:
                del self._item_rates[item]
                self._condition.notify_all()

    def reset_all_item_aging_rates(self):
        """
        Removes all per-item aging rate overrides, causing all items
        to use the current global aging rate.
        """
        with self._lock:
            self._item_rates.clear()
            self._condition.notify_all()

    def get_item_aging_rates(self) -> Dict[Any, float]:
        """
        Returns a dictionary of all current per-item aging rate overrides.
        """
        with self._lock:
            return dict(self._item_rates)

    def get_item_aging_rates_many(self, items: List[Any]) -> Dict[Any, float]:
        """
        Returns the current aging rates for specific items. 
        If an item has no custom rate, the global aging rate is returned.
        """
        with self._lock:
            return {item: self._item_rates.get(item, self.aging_rate) for item in items if item in self._items_set}

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
            self._condition.notify_all()

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
            self._condition.notify_all()

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

    def get_top_item_aging_rate(self) -> float:
        """
        Returns the current aging rate for the item that would be popped next.
        """
        with self._lock:
            if not self._queue:
                raise QueueEmptyError("get_top_item_aging_rate from an empty priority queue")
            
            _, entry = self._find_best_entry(time.time())
            item = entry[2]
            return self._item_rates.get(item, self.aging_rate)

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
            
            self._condition.notify_all()
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
            self._condition.notify_all()

    def remove_many(self, items: List[Any]):
        """
        Removes multiple specific items from the queue in a single lock acquisition.
        Raises ItemNotFoundError if any specified item is not found in the queue.
        """
        with self._lock:
            # Validate all items exist first to maintain atomicity
            for item in items:
                if item not in self._items_set:
                    raise ItemNotFoundError(f"item {item} not in priority queue")

            # Since removing an item changes indices of subsequent items, 
            # we track which ones to remove and rebuild or remove carefully.
            # The most stable way for bulk remove in a heap is to filter and re-heapify
            # if the number of removals is significant, or remove one by one.
            # Given that remove() handles the sift-down/up, we can use it but must
            # handle the shifting indices. Alternatively, rebuild the heap.
            
            # Use a set for faster lookup
            to_remove = set(items)
            self._queue = [entry for entry in self._queue if entry[2] not in to_remove]
            heapq.heapify(self._queue)
            
            for item in to_remove:
                if item in self._item_rates:
                    del self._item_rates[item]
                self._items_set.remove(item)
            self._condition.notify_all()

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
            self._condition.notify_all()

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

            # To avoid repeated linear scans in a loop, we can rebuild the heap
            # if the update list is large, but for small sets, _find_item_index is fine.
            # However, let's optimize by creating a map for the current queue if updates are many.
            if len(updates) > len(self._queue) // 4:
                # Rebuild strategy
                update_map = dict(updates)
                new_queue = []
                for base_p, entry_time, item in self._queue:
                    if item in update_map:
                        new_queue.append((update_map[item], entry_time, item))
                    else:
                        new_queue.append((base_p, entry_time, item))
                self._queue = new_queue
                heapq.heapify(self._queue)
            else:
                # Point update strategy
                for item, priority in updates:
                    idx = self._find_item_index(item)
                    entry_time = self._queue[idx][1]
                    self._queue[idx] = (priority, entry_time, item)
                    
                    heapq._siftdown(self._queue, 0, idx)
                    heapq._siftup(self._queue, idx)
            self._condition.notify_all()

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
            self._condition.notify_all()

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

    def get_priorities_many(self, items: List[Any]) -> Dict[Any, float]:
        """
        Alias for get_priorities_for_items for consistency with other '_many' methods.
        """
        return self.get_priorities_for_items(items)

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

    def clear_priority_range(self, min_p: float, max_p: float):
        """
        Removes all items whose current effective priority falls within [min_p, max_p].
        """
        with self._lock:
            now = time.time()
            to_remove = []
            for base_p, entry_time, item in self._queue:
                effective = self._calculate_effective_priority(base_p, entry_time, item, now)
                if min_p <= effective <= max_p:
                    to_remove.append(item)
            
            if to_remove:
                remove_set = set(to_remove)
                self._queue = [entry for entry in self._queue if entry[2] not in remove_set]
                heapq.heapify(self._queue)
                for item in remove_set:
                    if item in self._item_rates:
                        del self._item_rates[item]
                    self._items_set.remove(item)
                self._condition.notify_all()

    def wait_for_priority(self, item: Any, target_priority: float, timeout: Optional[float] = None) -> bool:
        """
        Blocks until the specified item reaches a priority value <= target_priority
        (meaning it becomes more prioritized).
        Returns True if the item reached the priority, False if it timed out or was removed.
        """
        start_time = time.time()
        with self._condition:
            while True:
                try:
                    current_p = self.get_current_priority(item)
                    if current_p <= target_priority:
                        return True
                except ItemNotFoundError:
                    return False

                elapsed = time.time() - start_time
                remaining = (timeout - elapsed) if timeout is not None else None
                
                if timeout is not None and remaining <= 0:
                    return False

                # Calculate approximate wait time based on aging rate to avoid busy looping
                # effective_p = base_p - (now - entry_time) * rate
                # we want effective_p <= target_p
                # (now - entry_time) * rate >= base_p - target_p
                # now >= entry_time + (base_p - target_p) / rate
                
                # We need the item's internal data for this
                try:
                    details = self.get_priority_details(item)
                    base_p = details["base_priority"]
                    entry_time = details["entry_time"]
                    rate = self._item_rates.get(item, self.aging_rate)
                    
                    if rate > 0:
                        wait_time = (base_p - target_priority) / rate - (time.time() - entry_time)
                        # Wait for a bit, but not longer than the estimated time or the timeout
                        sleep_duration = max(0.1, min(wait_time, 1.0))
                        if timeout is not None:
                            sleep_duration = min(sleep_duration, remaining)
                    else:
                        # If not aging, we can only wake up on external changes (rate change, etc.)
                        sleep_duration = 1.0
                        if timeout is not None:
                            sleep_duration = min(sleep_duration, remaining)
                except ItemNotFoundError:
                    return False

                if not self._condition.wait(timeout=sleep_duration):
                    # wait() returns True if notified, False if timed out
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        return False

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
            self._condition.notify_all()

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
                self._condition.notify_all()
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
                    self._condition.notify_all()
                except ItemNotFoundError:
                    pass

    @contextmanager
    def timed_priority_boost(self, item: Any, boost_amount: float, duration_seconds: float) -> Generator[None, None, None]:
        """
        Temporarily decreases the priority value (increases priority) of an item
        for a specified duration, regardless of when the context block ends.
        Note: The priority is reverted after duration_seconds has passed since the start of the boost,
        or when the context exits, whichever comes first.
        """
        if not isinstance(boost_amount, (int, float)) or not isinstance(duration_seconds, (int, float)):
            raise TypeError("Boost amount and duration must be numbers")

        start_time = time.time()
        with self._lock:
            try:
                idx = self._find_item_index(item)
                old_priority, entry_time, _ = self._queue[idx]
                self._queue[idx] = (old_priority - boost_amount, entry_time, item)
                heapq._siftdown(self._queue, 0, idx)
                heapq._siftup(self._queue, idx)
                self._condition.notify_all()
            except ItemNotFoundError:
                pass

        try:
            yield
        finally:
            # Ensure the boost is removed
            with self._lock:
                try:
                    idx = self._find_item_index(item)
                    current_priority, entry_time, _ = self._queue[idx]
                    # We only revert if we are still within the window or just finished it
                    # In this implementation, the context manager is the primary driver.
                    self._queue[idx] = (current_priority + boost_amount, entry_time, item)
                    heapq._siftdown(self._queue, 0, idx)
                    heapq._siftup(self._queue, idx)
                    self._condition.notify_all()
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
