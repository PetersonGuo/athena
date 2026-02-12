"""Tests for cuda_memory.py."""

from __future__ import annotations

from athena.memory.cuda_memory import CudaMemoryProfiler, _format_bytes


def test_cuda_not_available():
    """Test graceful degradation when CUDA is not available."""
    cm = CudaMemoryProfiler()
    # On most test machines, CUDA won't be available
    if not cm.is_available():
        result = cm.get_memory_stats()
        assert "error" in result

        result = cm.get_memory_summary()
        assert "error" in result

        result = cm.get_live_tensors()
        assert "error" in result

        result = cm.take_tensor_snapshot("test")
        assert "error" in result


def test_format_bytes():
    assert _format_bytes(0) == "0 B"
    assert "GiB" in _format_bytes(2 * 1024**3)
