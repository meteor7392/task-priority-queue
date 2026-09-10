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

    def test_empty_pop(self):
        pq = AgingPriorityQueue()
        with self.assertRaises(IndexError):
            pq.pop()

if __name__ == "__main__":
    unittest.main()