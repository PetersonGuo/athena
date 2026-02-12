"""PyTorch CUDA memory profiling. Gracefully no-ops if torch is unavailable."""

from __future__ import annotations

import gc
import sys
from typing import Any


class CudaMemoryProfiler:
    """PyTorch CUDA memory profiling.

    All methods return informative error dicts if torch/CUDA is unavailable.
    """

    def __init__(self):
        self._torch_available = False
        self._cuda_available = False
        self._torch = None
        try:
            import torch
            self._torch = torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        self._tensor_snapshots: list[tuple[str, list[dict[str, Any]]]] = []

    def is_available(self) -> bool:
        return self._cuda_available

    def _check_available(self) -> dict[str, str] | None:
        if not self._torch_available:
            return {"error": "PyTorch is not installed"}
        if not self._cuda_available:
            return {"error": "CUDA is not available"}
        return None

    def get_memory_stats(self, device: int | None = None) -> dict[str, Any]:
        """Return allocated, max_allocated, reserved, max_reserved (bytes)."""
        err = self._check_available()
        if err:
            return err

        torch = self._torch
        dev = device or 0
        allocated = torch.cuda.memory_allocated(dev)
        max_allocated = torch.cuda.max_memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        max_reserved = torch.cuda.max_memory_reserved(dev)

        return {
            "device": dev,
            "allocated_bytes": allocated,
            "allocated_human": _format_bytes(allocated),
            "max_allocated_bytes": max_allocated,
            "max_allocated_human": _format_bytes(max_allocated),
            "reserved_bytes": reserved,
            "reserved_human": _format_bytes(reserved),
            "max_reserved_bytes": max_reserved,
            "max_reserved_human": _format_bytes(max_reserved),
        }

    def get_memory_summary(self, device: int | None = None) -> dict[str, str]:
        """Return torch.cuda.memory_summary() as a string."""
        err = self._check_available()
        if err:
            return err

        dev = device or 0
        summary = self._torch.cuda.memory_summary(dev)
        return {"device": dev, "summary": summary}

    def get_live_tensors(
        self,
        device: int | None = None,
        sort_by: str = "size",
        top_n: int = 30,
    ) -> dict[str, Any]:
        """Scan gc.get_objects() for CUDA tensors."""
        err = self._check_available()
        if err:
            return err

        torch = self._torch
        tensors = []

        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    if device is not None and obj.device.index != device:
                        continue
                    size_bytes = obj.nelement() * obj.element_size()
                    tensors.append({
                        "id": id(obj),
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "device": str(obj.device),
                        "size_bytes": size_bytes,
                        "size_human": _format_bytes(size_bytes),
                        "refcount": sys.getrefcount(obj),
                        "requires_grad": obj.requires_grad,
                        "is_leaf": obj.is_leaf,
                    })
            except Exception:
                continue

        if sort_by == "size":
            tensors.sort(key=lambda t: t["size_bytes"], reverse=True)
        elif sort_by == "refcount":
            tensors.sort(key=lambda t: t["refcount"], reverse=True)

        total_size = sum(t["size_bytes"] for t in tensors)

        return {
            "total_tensors": len(tensors),
            "total_size_bytes": total_size,
            "total_size_human": _format_bytes(total_size),
            "tensors": tensors[:top_n],
        }

    def take_tensor_snapshot(self, label: str) -> dict[str, Any]:
        """Snapshot all live CUDA tensors for later comparison."""
        result = self.get_live_tensors()
        if "error" in result:
            return result

        self._tensor_snapshots.append((label, result["tensors"]))
        return {
            "label": label,
            "total_tensors": result["total_tensors"],
            "total_size_human": result["total_size_human"],
            "snapshot_index": len(self._tensor_snapshots) - 1,
        }

    def compare_tensor_snapshots(
        self,
        label_a: str,
        label_b: str,
    ) -> dict[str, Any]:
        """Compare two tensor snapshots to identify new/removed tensors."""
        snap_a = self._find_snapshot(label_a)
        snap_b = self._find_snapshot(label_b)
        if snap_a is None:
            return {"error": f"Snapshot not found: {label_a}"}
        if snap_b is None:
            return {"error": f"Snapshot not found: {label_b}"}

        ids_a = {t["id"] for t in snap_a}
        ids_b = {t["id"] for t in snap_b}

        new_ids = ids_b - ids_a
        removed_ids = ids_a - ids_b
        persistent_ids = ids_a & ids_b

        new_tensors = [t for t in snap_b if t["id"] in new_ids]
        removed_tensors = [t for t in snap_a if t["id"] in removed_ids]

        new_size = sum(t["size_bytes"] for t in new_tensors)
        removed_size = sum(t["size_bytes"] for t in removed_tensors)

        return {
            "label_a": label_a,
            "label_b": label_b,
            "new_count": len(new_tensors),
            "new_size_human": _format_bytes(new_size),
            "removed_count": len(removed_tensors),
            "removed_size_human": _format_bytes(removed_size),
            "persistent_count": len(persistent_ids),
            "new_tensors": new_tensors[:20],
            "removed_tensors": removed_tensors[:20],
        }

    def get_snapshot_labels(self) -> list[str]:
        return [label for label, _ in self._tensor_snapshots]

    def export_snapshots(self) -> list[dict[str, Any]]:
        """Export all tensor snapshots as serializable dicts for checkpointing."""
        return [
            {"label": label, "tensors": tensors}
            for label, tensors in self._tensor_snapshots
        ]

    def restore_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        """Restore tensor snapshots from checkpoint data."""
        self._tensor_snapshots = [
            (entry["label"], entry["tensors"])
            for entry in snapshots
            if "label" in entry and "tensors" in entry
        ]

    def _find_snapshot(self, label: str) -> list[dict[str, Any]] | None:
        for lbl, snap in self._tensor_snapshots:
            if lbl == label:
                return snap
        return None


def _format_bytes(size: int | float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024:
            if unit == "B":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
