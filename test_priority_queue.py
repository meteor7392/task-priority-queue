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

    def test_push_many(self):
        pq = AgingPriorityQueue(aging_rate=0)
        tasks = [(10, "Low"), (1, "High"), (5, "Medium")]
        pq.push_many(tasks)
        self.assertEqual(len(pq), 3)
        self.assertEqual(pq.pop(), "High")
        self.assertEqual(pq.pop(), "Medium")
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

    def test_priority_boost(self):
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Low")
        pq.push(5, "Medium")
        
        # Initially Medium is highest
        self.assertEqual(pq.peek(), "Medium")
        
        with pq.priority_boost("Low", 10):
            # Low priority (10) boosted by 10 = 0, now highest
            self.assertEqual(pq.peek(), "Low")
            
        # After context, should return to Medium
        self.assertEqual(pq.peek(), "Medium")

    def test_get_sorted_tasks(self):
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Low")
        pq.push(1, "High")
        pq.push(5, "Medium")
        
        self.assertEqual(pq.get_sorted_tasks(), ["High", "Medium", "Low"])
        
        # Test with aging
        pq_aging = AgingPriorityQueue(aging_rate=10)
        pq_aging.push(20, "Low")
        pq_aging.push(5, "High")
        # Wait until Low (20) aged by 20 becomes 0, making it higher priority than High (5)
        time.sleep(2.1)
        self.assertEqual(pq_aging.get_sorted_tasks()[0], "Low")

    def test_aging_reorder_pop(self):
        """Verify that pop() correctly identifies the aged highest priority item."""
        pq_gap = AgingPriorityQueue(aging_rate=10)
        pq_gap.push(20, "A") # Base 20
        time.sleep(2.0)       # A ages by 20
        pq_gap.push(10, "B") # Base 10
        # At this moment: A is 20 - 20 = 0. B is 10. A should be popped.
        self.assertEqual(pq_gap.pop(), "A")

    def test_set_aging_rate(self):
        """Verify that changing the aging rate mid-flight affects priorities."""
        pq = AgingPriorityQueue(aging_rate=0)
        pq.push(10, "Low")
        pq.push(5, "High")
        
        # Currently High (5) is top
        self.assertEqual(pq.peek(), "High")
        
        # Change aging rate to be very high
        pq.set_aging_rate(100)
        time.sleep(0.1) # Age low priority task significantly
        
        # Now Low should have aged enough to beat High
        # Low: 10 - (0.1 * 100) = 0
        # High: 5 - (0.1 * 100) = -5 
        # Wait, if both age, the relative order stays same unless we push after rate change
        # Let's try a different scenario: push Low, wait, push High, then change rate.
        
        pq.clear()
        pq.set_aging_rate(0)
        pq.push(20, "OldLow")
        time.sleep(0.5)
        pq.push(10, "NewHigh")
        
        # NewHigh is better (10 < 20)
        self.assertEqual(pq.peek(), "NewHigh")
        
        # Boost aging rate so OldLow catches up
        pq.set_aging_rate(100)
        # OldLow has been there for ~0.5s. Now it ages by 100/s
        # Effective priority: 20 - (0.5 * 100) = -30
        # NewHigh effective: 10 - (0 * 100) approx = 10
        self.assertEqual(pq.peek(), "OldLow")

if __name__ == "__main__":
    unittest.main()