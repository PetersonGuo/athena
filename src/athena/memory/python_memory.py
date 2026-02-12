"""Python memory profiling using tracemalloc and gc."""

from __future__ import annotations

import gc
import sys
import tracemalloc
from collections import Counter
from typing import Any


class PythonMemoryProfiler:
    """Python-level memory profiling.

    Capabilities:
    - tracemalloc: start/stop tracing, snapshots, comparison
    - gc: object counts, generation stats, reference analysis
    """

    def __init__(self, nframes: int = 25):
        self._nframes = nframes
        self._snapshots: list[tuple[str, tracemalloc.Snapshot]] = []
        self._tracing = False

    def start_tracing(self, nframes: int | None = None) -> dict[str, Any]:
        """Start tracemalloc tracing."""
        if self._tracing:
            return {"status": "already_tracing"}
        nf = nframes or self._nframes
        tracemalloc.start(nf)
        self._tracing = True
        return {"status": "started", "nframes": nf}

    def stop_tracing(self) -> dict[str, Any]:
        """Stop tracemalloc tracing."""
        if not self._tracing:
            return {"status": "not_tracing"}
        tracemalloc.stop()
        self._tracing = False
        return {"status": "stopped"}

    def take_snapshot(self, label: str = "") -> dict[str, Any]:
        """Take a tracemalloc snapshot and store it."""
        if not self._tracing:
            self.start_tracing()

        snapshot = tracemalloc.take_snapshot()
        # Filter out our own modules
        snapshot = snapshot.filter_traces([
            tracemalloc.Filter(False, "*/athena/*"),
            tracemalloc.Filter(False, "*/openai/*"),
            tracemalloc.Filter(False, "<frozen *>"),
        ])

        if not label:
            label = f"snapshot_{len(self._snapshots)}"
        self._snapshots.append((label, snapshot))

        stats = snapshot.statistics("lineno")
        total_size = sum(s.size for s in stats)

        return {
            "label": label,
            "total_size_bytes": total_size,
            "total_size_human": _format_bytes(total_size),
            "num_traces": len(stats),
            "snapshot_index": len(self._snapshots) - 1,
        }

    def compare_snapshots(
        self,
        label_a: str,
        label_b: str,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Compare two snapshots to see allocation changes."""
        snap_a = self._find_snapshot(label_a)
        snap_b = self._find_snapshot(label_b)
        if snap_a is None:
            return {"error": f"Snapshot not found: {label_a}"}
        if snap_b is None:
            return {"error": f"Snapshot not found: {label_b}"}

        diff = snap_b.compare_to(snap_a, "lineno")
        top = diff[:top_n]

        results = []
        for stat in top:
            results.append({
                "file": str(stat.traceback),
                "size_diff": stat.size_diff,
                "size_diff_human": _format_bytes(stat.size_diff),
                "size": stat.size,
                "size_human": _format_bytes(stat.size),
                "count_diff": stat.count_diff,
                "count": stat.count,
            })

        total_diff = sum(s.size_diff for s in diff)
        return {
            "label_a": label_a,
            "label_b": label_b,
            "total_diff_bytes": total_diff,
            "total_diff_human": _format_bytes(total_diff),
            "top_changes": results,
        }

    def get_top_allocations(
        self,
        top_n: int = 20,
        key_type: str = "lineno",
    ) -> list[dict[str, Any]]:
        """Return top N allocations from the most recent snapshot."""
        if not self._snapshots:
            if not self._tracing:
                self.start_tracing()
            self.take_snapshot("auto")

        _, snapshot = self._snapshots[-1]
        stats = snapshot.statistics(key_type)[:top_n]

        return [
            {
                "file": str(s.traceback),
                "size": s.size,
                "size_human": _format_bytes(s.size),
                "count": s.count,
            }
            for s in stats
        ]

    def get_current_memory(self) -> dict[str, Any]:
        """Return current and peak traced memory."""
        if not self._tracing:
            return {"error": "Tracing not active. Take a snapshot to start."}

        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_bytes": current,
            "current_human": _format_bytes(current),
            "peak_bytes": peak,
            "peak_human": _format_bytes(peak),
        }

    def gc_stats(self, top_n_types: int = 20) -> dict[str, Any]:
        """Get garbage collector statistics."""
        gc.collect()
        counts = gc.get_count()
        gc_stats = gc.get_stats()

        # Count objects by type
        type_counts = Counter(type(obj).__qualname__ for obj in gc.get_objects())
        top_types = type_counts.most_common(top_n_types)

        return {
            "generation_counts": {
                "gen0": counts[0],
                "gen1": counts[1],
                "gen2": counts[2],
            },
            "collection_stats": [
                {
                    "generation": i,
                    "collections": s.get("collections", 0),
                    "collected": s.get("collected", 0),
                    "uncollectable": s.get("uncollectable", 0),
                }
                for i, s in enumerate(gc_stats)
            ],
            "top_types": [
                {"type": t, "count": c} for t, c in top_types
            ],
            "total_objects": sum(type_counts.values()),
            "garbage_count": len(gc.garbage),
        }

    def get_referrers(self, obj: Any, max_results: int = 10) -> list[dict[str, str]]:
        """Find what objects refer to the given object."""
        referrers = gc.get_referrers(obj)
        results = []
        for ref in referrers[:max_results]:
            if ref is not obj:
                results.append({
                    "type": type(ref).__qualname__,
                    "repr": repr(ref)[:200],
                    "id": str(id(ref)),
                })
        return results

    def get_referents(self, obj: Any, max_results: int = 10) -> list[dict[str, str]]:
        """Find what objects the given object refers to."""
        referents = gc.get_referents(obj)
        results = []
        for ref in referents[:max_results]:
            results.append({
                "type": type(ref).__qualname__,
                "repr": repr(ref)[:200],
                "id": str(id(ref)),
            })
        return results

    def get_snapshot_labels(self) -> list[str]:
        """Return list of available snapshot labels."""
        return [label for label, _ in self._snapshots]

    def _find_snapshot(self, label: str) -> tracemalloc.Snapshot | None:
        for lbl, snap in self._snapshots:
            if lbl == label:
                return snap
        return None


def _format_bytes(size: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024:
            if unit == "B":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
