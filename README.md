# Task Priority Queue with Aging

A Python implementation of a priority queue that prevents low-priority task starvation by incrementally increasing the priority of tasks as they wait in the queue.

## Features
- Priority-based dequeuing
- Aging mechanism to prevent starvation
- Thread-safe operations
- Support for peeking, clearing, and querying current priority
- Dynamic priority updates

## Installation
```bash
pip install . # if packaged
```

## Usage
```python
from task_priority_queue import AgingPriorityQueue

# Initialize queue with an aging rate
queue = AgingPriorityQueue(aging_rate=1)

# Add tasks: (priority, data). Lower number = higher priority.
queue.push(10, "Low priority task")
queue.push(1, "High priority task")

# Check the current effective priority of a task
print(f"Current priority: {queue.get_current_priority('Low priority task')}")

# Update the base priority of a task
queue.update_priority("Low priority task", 2)

# After some time/operations, low priority tasks gain priority
print(queue.pop())

# Clear all tasks
queue.clear()
```