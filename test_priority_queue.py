import unittest
import time
from task_priority_queue import AgingPriorityQueue

class TestAgingPriorityQueue(unittest.TestCase):
    def test_basic_priority(self):
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Low")
        pq.push(1, "High")
        self.assertEqual(pq.pop(), "High")
        self.assertEqual(pq.pop(), "Low")

    def test_aging_mechanism(self):
        # High aging rate to make the effect immediate
        pq = AgingPriorityQueue(aging_rate=100)
        pq.push(100, "Very Low")
        pq.push(10, "Medium")
        
        # Wait enough for "Very Low" to jump ahead of "Medium"
        # (100 - (1 * 100)) = 0, which is < 10
        time.sleep(1.1)
        
        self.assertEqual(pq.pop(), "Very Low")

    def test_peek(self):
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Low")
        pq.push(1, "High")
        self.assertEqual(pq.peek(), "High")
        # Ensure peek didn't remove the item
        self.assertEqual(len(pq), 2)
        self.assertEqual(pq.pop(), "High")

    def test_empty_pop(self):
        pq = AgingPriorityQueue()
        with self.assertRaises(IndexError):
            pq.pop()

    def test_empty_peek(self):
        pq = AgingPriorityQueue()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_contains(self):
        pq = AgingPriorityQueue()
        pq.push(1, "Task A")
        pq.push(2, "Task B")
        self.assertIn("Task A", pq)
        self.assertIn("Task B", pq)
        self.assertNotIn("Task C", pq)

    def test_iteration(self):
        pq = AgingPriorityQueue()
        items = ["A", "B", "C"]
        for item in items:
            pq.push(1, item)
        
        iterated_items = list(pq)
        self.assertEqual(len(iterated_items), 3)
        for item in items:
            self.assertIn(item, iterated_items)

    def test_remove(self):
        pq = AgingPriorityQueue()
        pq.push(1, "Task A")
        pq.push(2, "Task B")
        pq.push(3, "Task C")
        
        pq.remove("Task B")
        self.assertNotIn("Task B", pq)
        self.assertEqual(len(pq), 2)
        
        with self.assertRaises(ValueError):
            pq.remove("Task D")

    def test_update_priority(self):
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Task A")
        pq.push(5, "Task B")
        
        # Initially Task B is highest priority
        self.assertEqual(pq.peek(), "Task B")
        
        # Update Task A to be higher priority than Task B
        pq.update_priority("Task A", 2)
        self.assertEqual(pq.peek(), "Task A")
        
        # Ensure update_priority raises ValueError for missing items
        with self.assertRaises(ValueError):
            pq.update_priority("Task C", 1)

if __name__ == "__main__":
    unittest.main()