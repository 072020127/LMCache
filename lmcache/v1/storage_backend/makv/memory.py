# SPDX-License-Identifier: Apache-2.0

"""MaKV in-memory object representation for delayed restore."""

# Standard
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObjMetadata

logger = init_logger(__name__)


class MaKVQuantizedMemoryObj(BytesBufferMemoryObj):
    """Parsed MaKV object kept compressed until GPU restore."""

    object_type = "makv_quantized"

    def __init__(
        self,
        raw_bytes: bytes,
        *,
        metadata_dict: dict[str, Any],
        payloads: dict[str, bytes | memoryview],
        protocol_version: int = 1,
    ) -> None:
        super().__init__(raw_bytes)
        self.makv_metadata = metadata_dict
        self.makv_payloads = payloads
        self.makv_protocol_version = int(protocol_version)
        self.meta = MemoryObjMetadata(
            shape=torch.Size([len(raw_bytes), 0, 0, 0]),
            dtype=torch.uint8,
            address=0,
            phy_size=len(raw_bytes),
            ref_count=1,
            pin_count=0,
            fmt=self.meta.fmt,
        )

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def qdm_metadata(self):
        """Return an explicitly requested shadow witness, when present.

        Production restore never calls this accessor.  The descriptor check
        also keeps legacy/QDM-off objects free of a QDM import or decode.
        """
        if (
            "qdm" not in self.makv_metadata
            and "qdm_version" not in self.makv_metadata
        ):
            return None
        from lmcache.v1.storage_backend.makv.qdm import load_qdm_metadata

        return load_qdm_metadata(self.makv_metadata, self.makv_payloads)
