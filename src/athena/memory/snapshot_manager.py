"""Coordinate Python and CUDA memory snapshots."""

from __future__ import annotations

from typing import Any

from athena.memory.cuda_memory import CudaMemoryProfiler
from athena.memory.python_memory import PythonMemoryProfiler


class SnapshotManager:
    """Coordinates memory snapshots across Python and CUDA subsystems."""

    def __init__(
        self,
        python_profiler: PythonMemoryProfiler,
        cuda_profiler: CudaMemoryProfiler,
    ):
        self._python = python_profiler
        self._cuda = cuda_profiler

    def take_combined_snapshot(self, label: str) -> dict[str, Any]:
        """Take both Python and CUDA snapshots with the same label."""
        result: dict[str, Any] = {"label": label}

        py_result = self._python.take_snapshot(label)
        result["python"] = py_result

        if self._cuda.is_available():
            cuda_result = self._cuda.take_tensor_snapshot(label)
            result["cuda"] = cuda_result

        return result

    def compare_combined(self, label_a: str, label_b: str) -> dict[str, Any]:
        """Compare combined snapshots."""
        result: dict[str, Any] = {
            "label_a": label_a,
            "label_b": label_b,
        }

        py_cmp = self._python.compare_snapshots(label_a, label_b)
        result["python"] = py_cmp

        if self._cuda.is_available():
            cuda_cmp = self._cuda.compare_tensor_snapshots(label_a, label_b)
            result["cuda"] = cuda_cmp

        return result
