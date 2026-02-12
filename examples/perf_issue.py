"""Example script with performance issues for testing perf profiling.

Run with: athena --perf-mode examples/perf_issue.py
"""

import time


def fibonacci_naive(n):
    """Extremely slow recursive fibonacci - O(2^n)."""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_fast(n):
    """Linear-time iterative fibonacci."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def sort_already_sorted(data):
    """Repeatedly sorts data that's already sorted."""
    result = data
    for _ in range(50):
        result = sorted(result)
    return result


def build_string_slowly(n):
    """Builds a large string via concatenation instead of join - O(n^2)."""
    result = ""
    for i in range(n):
        result = result + f"item-{i}," + " " * 100
    return result


def build_string_fast(n):
    """Builds a large string via join - O(n)."""
    return "".join(f"item-{i}," + " " * 100 for i in range(n))


def search_linearly(data, targets):
    """Searches a list for each target - O(n*m) instead of O(n+m) with a set."""
    found = []
    for target in targets:
        if target in data:
            found.append(target)
    return found


def process_data(records):
    """Processes records with multiple hidden performance issues."""
    sorted_records = sort_already_sorted(records)

    report = build_string_slowly(len(sorted_records))

    ids = [r for r in range(len(sorted_records))]
    search_targets = list(range(0, len(sorted_records), 3))
    matches = search_linearly(ids, search_targets)

    return {
        "sorted_count": len(sorted_records),
        "report_length": len(report),
        "matches": len(matches),
    }


def main():
    print("Starting performance test...")

    # Phase 1: Slow fibonacci
    print("\nPhase 1: Computing fibonacci numbers...")
    start = time.time()
    results = []
    for n in [10, 20, 25, 30, 33]:
        val = fibonacci_naive(n)
        elapsed = time.time() - start
        print(f"  fib({n}) = {val} (cumulative: {elapsed:.2f}s)")
        results.append(val)

    # Phase 2: Data processing with hidden issues
    print("\nPhase 2: Processing data...")
    data = list(range(10_000))
    start = time.time()
    stats = process_data(data)
    elapsed = time.time() - start
    print(f"  Processed {stats['sorted_count']} records in {elapsed:.2f}s")
    print(f"  Report length: {stats['report_length']}")
    print(f"  Matches found: {stats['matches']}")

    # Phase 3: Compare fast vs slow
    print("\nPhase 3: Fast vs slow string building...")
    n = 50_000
    start = time.time()
    slow = build_string_slowly(n)
    slow_time = time.time() - start
    print(f"  Slow (concatenation): {slow_time:.3f}s, length={len(slow)}")

    start = time.time()
    fast = build_string_fast(n)
    fast_time = time.time() - start
    print(f"  Fast (join):          {fast_time:.3f}s, length={len(fast)}")
    print(f"  Speedup: {slow_time / max(fast_time, 0.0001):.1f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
