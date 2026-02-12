"""Example script with a memory leak for testing memory profiling.

Run with: athena --trace-memory examples/memory_leak.py
"""

import time


class DataProcessor:
    """Processor that accumulates results."""

    def __init__(self):
        self.history = []
        self._cache = {}

    def process(self, data):
        result = [x * 2 for x in data]
        self.history.append(result)
        self._cache[id(data)] = result
        return result


def main():
    processor = DataProcessor()

    for i in range(100):
        # Generate increasingly large data
        data = list(range(i * 1000))
        result = processor.process(data)

        if i % 10 == 0:
            print(f"Iteration {i}: history size = {len(processor.history)}")

    print(f"Final history size: {len(processor.history)}")
    print(f"Cache size: {len(processor._cache)}")


if __name__ == "__main__":
    main()
