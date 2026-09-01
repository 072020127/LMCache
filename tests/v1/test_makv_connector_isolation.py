# SPDX-License-Identifier: Apache-2.0

"""Regression tests for mode-independent GPU connector imports."""

# Standard
import os
from pathlib import Path
import subprocess
import sys


def test_cachegen_connector_does_not_import_makv_restore() -> None:
    """Importing the common connector must not load MaKV implementation code."""
    repo_root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    code = """
import sys
import lmcache.v1.gpu_connector.gpu_connectors
import lmcache.v1.storage_backend.naive_serde

assert "lmcache.v1.storage_backend.makv.memory" not in sys.modules
assert "lmcache.v1.storage_backend.makv.paged_restore" not in sys.modules
assert "lmcache.v1.storage_backend.makv.serde" not in sys.modules
assert "lmcache.v1.storage_backend.makv.quantizer" not in sys.modules
assert "lmcache.v1.storage_backend.makv.scout_overlap" not in sys.modules
assert "lmcache.v1.gpu_connector.makv_restore" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_makv_object_type_is_explicit() -> None:
    """The MaKV branch can be selected without an isinstance import."""
    from lmcache.v1.gpu_connector.makv_restore import (
        is_makv_quantized_memory_obj,
    )

    class Marker:
        object_type = "makv_quantized"

    assert is_makv_quantized_memory_obj(Marker())


def test_native_serde_factory_does_not_load_makv() -> None:
    """Naive and CacheGen factory selection must remain MaKV-free."""
    repo_root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    code = """
import sys
from types import SimpleNamespace

import torch

from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.naive_serde import CreateSerde

metadata = LMCacheMetadata(
    model_name="test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.bfloat16,
    kv_shape=(32, 2, 8, 8, 128),
    chunk_size=8,
)
config = SimpleNamespace(chunk_size=8)
CreateSerde("naive", metadata, config)
CreateSerde("cachegen", metadata, config)
assert "lmcache.v1.storage_backend.makv.serde" not in sys.modules
assert "lmcache.v1.storage_backend.makv.quantizer" not in sys.modules
assert "lmcache.v1.gpu_connector.makv_restore" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_config_validation_does_not_load_makv() -> None:
    """Native config validation must not import MaKV-only configuration code."""
    repo_root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    code = """
import sys

from lmcache.v1.config import LMCacheEngineConfig

config = LMCacheEngineConfig.from_defaults(remote_serde="naive")
config.validate()
assert "lmcache.v1.storage_backend.makv.config" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
