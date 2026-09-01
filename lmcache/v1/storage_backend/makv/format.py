# SPDX-License-Identifier: Apache-2.0

"""Binary formats for MaKV PUT envelopes and stored objects."""

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import binascii
import json
import struct

# First Party
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan

CLIENT_MAGIC = b"MKVP"
OBJECT_MAGIC = b"MAKV"
HEADER_STRUCT = struct.Struct("<4sIIII")
CHECKSUM_OFFSET = 16
SUPPORTED_OBJECT_VERSIONS = (1,)
PAYLOAD_ALIGNMENT = 64


@dataclass(frozen=True)
class ClientPutEnvelope:
    key: str
    object_type: str
    metadata: dict[str, Any]
    raw_kv_payload: bytes


@dataclass(frozen=True)
class MaKVObject:
    object_type: str
    metadata: dict[str, Any]
    payloads: dict[str, bytes | memoryview]
    checksum: int
    protocol_version: int = 1


def peek_makv_header(blob: bytes | memoryview) -> tuple[int, int, int]:
    """Return ``(version, total_length, checksum)`` without scanning payloads."""
    if len(blob) < HEADER_STRUCT.size:
        raise ValueError("truncated MaKV blob header")
    magic, version, _, total_length, checksum = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != OBJECT_MAGIC:
        raise ValueError("not a MaKV object")
    if version not in SUPPORTED_OBJECT_VERSIONS:
        raise ValueError(f"Unsupported MaKV protocol version {version}")
    if total_length != len(blob):
        raise ValueError("MaKV blob length mismatch")
    return int(version), int(total_length), int(checksum)


def _build_metadata_with_offsets(
    metadata: dict[str, Any], payloads: dict[str, bytes]
) -> dict[str, Any]:
    wrapped_meta = dict(metadata)
    wrapped_meta["_payload_table"] = []
    while True:
        meta_bytes = json.dumps(
            wrapped_meta, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        offset = HEADER_STRUCT.size + len(meta_bytes)
        payload_table = []
        for name, data in payloads.items():
            offset = (offset + PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT
            offset *= PAYLOAD_ALIGNMENT
            entry: dict[str, Any] = {
                "name": name,
                "offset": offset,
                "length": len(data),
            }
            payload_table.append(entry)
            offset += len(data)
        new_meta = dict(metadata)
        new_meta["_payload_table"] = payload_table
        if new_meta == wrapped_meta:
            return wrapped_meta
        wrapped_meta = new_meta


def _encode_blob(
    magic: bytes,
    metadata: dict[str, Any],
    payloads: dict[str, bytes],
    *,
    version: int = 1,
) -> bytes:
    if version not in SUPPORTED_OBJECT_VERSIONS:
        raise ValueError(f"unsupported MaKV protocol version {version}")
    wrapped_meta = _build_metadata_with_offsets(metadata, payloads)
    meta_bytes = json.dumps(wrapped_meta, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    payload_table = {
        str(entry["name"]): entry
        for entry in wrapped_meta["_payload_table"]
    }
    total_len = HEADER_STRUCT.size + len(meta_bytes)
    for name, data in payloads.items():
        entry = payload_table[name]
        total_len = max(total_len, int(entry["offset"]) + len(data))
    body = bytearray(total_len)
    HEADER_STRUCT.pack_into(body, 0, magic, version, len(meta_bytes), total_len, 0)
    body[HEADER_STRUCT.size : HEADER_STRUCT.size + len(meta_bytes)] = meta_bytes
    for name, data in payloads.items():
        entry = payload_table[name]
        offset = int(entry["offset"])
        body[offset : offset + len(data)] = data
    checksum = binascii.crc32(body) & 0xFFFFFFFF
    struct.pack_into("<I", body, CHECKSUM_OFFSET, checksum)
    return bytes(body)


def _decode_blob(
    blob: bytes | memoryview,
    expected_magic: bytes,
    *,
    verify_checksum: bool = True,
) -> tuple[dict[str, Any], int, int]:
    if len(blob) < HEADER_STRUCT.size:
        raise ValueError("truncated MaKV blob header")
    magic, version, meta_len, total_len, checksum = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != expected_magic:
        raise ValueError(
            f"Unsupported MaKV blob magic {magic!r}, expected {expected_magic!r}"
        )
    if version not in SUPPORTED_OBJECT_VERSIONS:
        raise ValueError(f"Unsupported MaKV protocol version {version}")
    if total_len < HEADER_STRUCT.size or total_len != len(blob):
        raise ValueError("MaKV blob length mismatch")
    if meta_len > total_len - HEADER_STRUCT.size:
        raise ValueError("MaKV metadata length is out of bounds")
    if verify_checksum:
        # Feed the checksum field as zero without making a full-size copy of
        # the remote object. GETs are commonly hundreds of MiB per chunk.
        expected_checksum = binascii.crc32(blob[:CHECKSUM_OFFSET])
        expected_checksum = binascii.crc32(
            b"\x00\x00\x00\x00", expected_checksum
        )
        expected_checksum = binascii.crc32(
            blob[CHECKSUM_OFFSET + 4 :], expected_checksum
        ) & 0xFFFFFFFF
        if checksum != expected_checksum:
            raise ValueError("MaKV checksum mismatch")
    meta_start = HEADER_STRUCT.size
    meta_end = meta_start + meta_len
    try:
        metadata = json.loads(bytes(blob[meta_start:meta_end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid MaKV metadata JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("MaKV metadata must be an object")
    return metadata, meta_end, int(version)


def encode_client_put_envelope(
    *,
    key: str,
    object_type: str,
    plan: Optional[MaKVQuantPlan],
    raw_kv_payload: bytes,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> bytes:
    """Encode client PUT bytes for the MaKV connector."""
    metadata = {
        "key": key,
        "object_type": object_type,
        "plan": plan.to_dict() if plan is not None else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return _encode_blob(CLIENT_MAGIC, metadata, {"raw_kv_payload": raw_kv_payload})


def decode_client_put_envelope(blob: bytes) -> ClientPutEnvelope:
    """Decode a client PUT envelope."""
    metadata, meta_end, version = _decode_blob(blob, CLIENT_MAGIC)
    if version != 1:
        raise ValueError("client MaKV PUT envelopes only support protocol v1")
    table = metadata.pop("_payload_table", None)
    if not isinstance(table, list):
        raise ValueError("MaKV client envelope is missing its payload table")
    payloads = {}
    ranges: list[tuple[int, int]] = []
    for entry in table:
        offset = int(entry["offset"])
        length = int(entry["length"])
        if offset < meta_end or length < 0 or offset + length > len(blob):
            raise ValueError("MaKV client payload table out of bounds")
        ranges.append((offset, offset + length))
        payloads[entry["name"]] = blob[offset : offset + length]
    if len(payloads) != len(table) or _ranges_overlap(ranges):
        raise ValueError("MaKV client payload table overlaps or repeats entries")
    return ClientPutEnvelope(
        key=str(metadata["key"]),
        object_type=str(metadata["object_type"]),
        metadata=metadata,
        raw_kv_payload=payloads["raw_kv_payload"],
    )


def encode_makv_object(
    *,
    object_type: str,
    metadata: dict[str, Any],
    payloads: dict[str, bytes],
    protocol_version: int = 1,
) -> bytes:
    """Encode a stored MaKV object."""
    storage_meta = {
        "object_type": object_type,
        **metadata,
    }
    return _encode_blob(
        OBJECT_MAGIC,
        storage_meta,
        payloads,
        version=protocol_version,
    )


def _ranges_overlap(ranges: list[tuple[int, int]]) -> bool:
    previous_end = -1
    for start, end in sorted(ranges):
        if start < previous_end:
            return True
        previous_end = end
    return False


def decode_makv_object(
    blob: bytes,
    *,
    copy_payloads: bool = True,
    verify_checksum: bool = True,
) -> MaKVObject:
    """Decode a stored MaKV object, optionally retaining zero-copy views."""
    metadata, meta_end, version = _decode_blob(
        blob, OBJECT_MAGIC, verify_checksum=verify_checksum
    )
    table = metadata.pop("_payload_table", None)
    if not isinstance(table, list):
        raise ValueError("MaKV object is missing its payload table")
    payloads: dict[str, bytes | memoryview] = {}
    retained_table = []
    ranges: list[tuple[int, int]] = []
    for entry in table:
        try:
            offset = int(entry["offset"])
            length = int(entry["length"])
            name = str(entry["name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid MaKV payload table entry") from error
        if offset < meta_end or length < 0 or offset + length > len(blob):
            raise ValueError("MaKV payload table out of bounds")
        ranges.append((offset, offset + length))
        retained_entry = dict(entry)
        retained_entry.update({"name": name, "offset": offset, "length": length})
        retained_table.append(retained_entry)
        if copy_payloads:
            payloads[name] = blob[offset : offset + length]
        else:
            payloads[name] = memoryview(blob)[offset : offset + length]
    if len(payloads) != len(table) or _ranges_overlap(ranges):
        raise ValueError("MaKV payload table overlaps or repeats entries")
    metadata["_payload_table"] = retained_table
    checksum = int(metadata.get("checksum", 0))
    return MaKVObject(
        object_type=str(metadata.pop("object_type")),
        metadata=metadata,
        payloads=payloads,
        checksum=checksum,
        protocol_version=version,
    )
