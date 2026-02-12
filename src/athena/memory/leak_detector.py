"""Heuristic-based memory leak detection by analyzing snapshot series."""

from __future__ import annotations

import gc
from typing import Any

from athena.memory.cuda_memory import CudaMemoryProfiler
from athena.memory.python_memory import PythonMemoryProfiler


class LeakDetector:
    """Analyzes memory snapshot series to identify potential leaks.

    Heuristics:
    1. Monotonic allocation growth at the same source line
    2. Object type count growing monotonically
    3. CUDA tensor accumulation
    4. Reference cycles in gc.garbage
    """

    def __init__(
        self,
        python_profiler: PythonMemoryProfiler,
        cuda_profiler: CudaMemoryProfiler,
    ):
        self._python = python_profiler
        self._cuda = cuda_profiler

    def analyze(self) -> dict[str, Any]:
        """Analyze all collected snapshots and return suspected leaks."""
        suspected: list[dict[str, Any]] = []

        # Check for reference cycles
        gc.collect()
        if gc.garbage:
            suspected.append({
                "type": "reference_cycle",
                "description": f"{len(gc.garbage)} uncollectable objects found in gc.garbage",
                "count": len(gc.garbage),
                "sample_types": list({type(o).__qualname__ for o in gc.garbage[:10]}),
                "confidence": "high",
                "suggestion": (
                    "These objects have __del__ methods and are in reference cycles. "
                    "Consider removing __del__ or breaking the cycles."
                ),
            })

        # Compare Python snapshots if we have at least 2
        labels = self._python.get_snapshot_labels()
        if len(labels) >= 2:
            cmp = self._python.compare_snapshots(labels[0], labels[-1])
            if "top_changes" in cmp:
                for change in cmp["top_changes"][:5]:
                    if change["size_diff"] > 100_000:  # >100KB growth
                        suspected.append({
                            "type": "python_allocation_growth",
                            "location": change["file"],
                            "growth": change["size_diff_human"],
                            "current_size": change["size_human"],
                            "count_diff": change["count_diff"],
                            "confidence": "medium",
                            "suggestion": (
                                "This allocation grew significantly between snapshots. "
                                "Check if objects at this location are being retained "
                                "unnecessarily."
                            ),
                        })

        # Compare CUDA snapshots if available
        if self._cuda.is_available():
            cuda_labels = self._cuda.get_snapshot_labels()
            if len(cuda_labels) >= 2:
                cmp = self._cuda.compare_tensor_snapshots(
                    cuda_labels[0], cuda_labels[-1]
                )
                if "error" not in cmp and cmp.get("new_count", 0) > 0:
                    suspected.append({
                        "type": "cuda_tensor_accumulation",
                        "new_tensors": cmp["new_count"],
                        "new_size": cmp["new_size_human"],
                        "confidence": "medium",
                        "suggestion": (
                            "New CUDA tensors appeared since the first snapshot. "
                            "Check for tensors retained in computation graphs "
                            "(.detach()), accumulated in lists, or stored in "
                            "class attributes without being freed."
                        ),
                    })

        if not suspected:
            return {
                "status": "no_leaks_detected",
                "message": (
                    "No obvious memory leaks detected. "
                    "Take more snapshots at different points for better analysis."
                ),
                "snapshots_analyzed": len(labels),
            }

        return {
            "status": "potential_leaks_found",
            "suspected_leaks": suspected,
            "snapshots_analyzed": len(labels),
        }
