# SPDX-License-Identifier: Apache-2.0

"""Lazy boundary between the common GPU connector and MaKV."""

# Standard
import importlib
from typing import Any


def is_makv_quantized_memory_obj(memory_obj: Any) -> bool:
    """Detect a MaKV object without importing MaKV on the normal path."""
    if getattr(memory_obj, "object_type", None) == "makv_quantized":
        return True

    # Compatibility with objects created before the explicit object type was
    # added. This is intentionally structural and does not import MaKV.
    cls = type(memory_obj)
    return (
        cls.__name__ == "MaKVQuantizedMemoryObj"
        and cls.__module__.startswith("lmcache.v1.storage_backend.makv")
    )


def restore_makv_quantized_to_paged(
    memory_obj: Any, *args: Any, **kwargs: Any
) -> Any:
    """Dispatch MaKV restore only after the MaKV branch was selected."""
    module = importlib.import_module(
        "lmcache.v1.storage_backend.makv.paged_restore"
    )
    restore = module.restore_makv_quantized_to_paged
    return restore(memory_obj, *args, **kwargs)


def begin_makv_restore_timing_scope() -> int:
    """Create a request-local MaKV timing scope without eager MaKV imports."""
    module = importlib.import_module("lmcache.v1.storage_backend.makv.metrics")
    return module.RESTORE_METRICS.begin_restore_scope()


def finish_makv_restore_timing_scope(scope_id: int) -> dict[str, Any]:
    """Return a completed scope as plain data for cache-engine reporting."""
    module = importlib.import_module("lmcache.v1.storage_backend.makv.metrics")
    return module.RESTORE_METRICS.finish_restore_scope(scope_id).__dict__.copy()
