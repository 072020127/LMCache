# SPDX-License-Identifier: Apache-2.0

"""Length-delimited wire protocol for the MaKV Remote Manager."""

# Standard
from typing import Any
import asyncio
import json
import struct

FRAME_HEADER = struct.Struct("!IQ")
MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024

# GET_BATCH v1 carries one binary directory followed by the complete MaKV
# objects.  The objects remain independently checksummed/self-describing; the
# batch envelope only removes per-object wire framing and Python header work.
BATCH_BLOB_MAGIC = b"MKVB"
BATCH_BLOB_VERSION = 1
# GET_BATCH stream_v1 uses one regular length-delimited frame per object.
BATCH_STREAM_VERSION = 1
BATCH_BLOB_HEADER = struct.Struct("!4sIIQ")
BATCH_BLOB_ENTRY = struct.Struct("!B7xQQ")
BATCH_BLOB_ALIGNMENT = 64
MAX_BATCH_OBJECTS = 4096
PRECISION_RISK_OPERATION = "PRECISION_RISK"


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> tuple[dict[str, Any], bytes]:
    """Read and validate one manager protocol frame."""
    raw_header = await reader.readexactly(FRAME_HEADER.size)
    header_len, payload_len = FRAME_HEADER.unpack(raw_header)
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError("invalid MaKV frame header length")
    if payload_len > max_payload_bytes:
        raise ValueError("MaKV frame payload exceeds configured limit")
    header = json.loads((await reader.readexactly(header_len)).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("MaKV frame header must be an object")
    payload = await reader.readexactly(payload_len)
    return header, payload


async def write_frame(
    writer: asyncio.StreamWriter,
    header: dict[str, Any],
    payload: bytes = b"",
    *,
    drain: bool = True,
) -> None:
    """Write one manager protocol frame.

    Batch responses can queue several independent frames and drain once. This
    keeps each object self-delimiting without forcing the manager to concatenate
    large MaKV blobs into another full-size temporary buffer.
    """
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("MaKV frame header exceeds configured limit")
    writer.write(FRAME_HEADER.pack(len(header_bytes), len(payload)))
    writer.write(header_bytes)
    writer.write(payload)
    if drain:
        await writer.drain()


def _batch_blob_layout(
    values: list[bytes | bytearray | memoryview | None],
) -> tuple[bytearray, list[bytes | bytearray | memoryview], int]:
    """Build the small batch directory and its zero-copy payload segments."""
    count = len(values)
    if count > MAX_BATCH_OBJECTS:
        raise ValueError("MaKV batch blob contains too many objects")
    directory_end = BATCH_BLOB_HEADER.size + count * BATCH_BLOB_ENTRY.size
    offsets: list[tuple[int, int] | None] = []
    payload_offset = directory_end
    for value in values:
        if value is None:
            offsets.append(None)
            continue
        payload_offset = (
            (payload_offset + BATCH_BLOB_ALIGNMENT - 1)
            // BATCH_BLOB_ALIGNMENT
            * BATCH_BLOB_ALIGNMENT
        )
        length = len(value)
        offsets.append((payload_offset, length))
        payload_offset += length

    directory = bytearray(directory_end)
    BATCH_BLOB_HEADER.pack_into(
        directory, 0, BATCH_BLOB_MAGIC, BATCH_BLOB_VERSION, count, payload_offset
    )
    segments: list[bytes | bytearray | memoryview] = [directory]
    written_offset = directory_end
    for index, (value, offset_info) in enumerate(zip(values, offsets, strict=True)):
        entry_offset = BATCH_BLOB_HEADER.size + index * BATCH_BLOB_ENTRY.size
        if value is None or offset_info is None:
            BATCH_BLOB_ENTRY.pack_into(directory, entry_offset, 0, 0, 0)
            continue
        object_offset, length = offset_info
        BATCH_BLOB_ENTRY.pack_into(
            directory, entry_offset, 1, object_offset, length
        )
        padding = object_offset - written_offset
        if padding:
            segments.append(b"\x00" * padding)
        segments.append(value)
        written_offset = object_offset + length
    return directory, segments, payload_offset


def batch_blob_size(values: list[bytes | bytearray | memoryview | None]) -> int:
    """Return the encoded MKVB payload length without copying object bytes."""
    count = len(values)
    if count > MAX_BATCH_OBJECTS:
        raise ValueError("MaKV batch blob contains too many objects")
    payload_offset = BATCH_BLOB_HEADER.size + count * BATCH_BLOB_ENTRY.size
    for value in values:
        if value is None:
            continue
        payload_offset = (
            (payload_offset + BATCH_BLOB_ALIGNMENT - 1)
            // BATCH_BLOB_ALIGNMENT
            * BATCH_BLOB_ALIGNMENT
        ) + len(value)
    return payload_offset


async def write_batch_blob_frame(
    writer: asyncio.StreamWriter,
    header: dict[str, Any],
    values: list[bytes | bytearray | memoryview | None],
    *,
    drain: bool = True,
) -> None:
    """Write one MKVB frame without copying the complete object payload.

    TCP preserves the segment order, so the receiver observes one contiguous
    frame.  ``writer.write`` retains the existing object buffers until the
    drain completes; only the directory and alignment padding are allocated.
    """
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("MaKV frame header exceeds configured limit")
    directory, segments, payload_length = _batch_blob_layout(values)
    if payload_length != batch_blob_size(values):
        raise RuntimeError("MaKV batch blob layout size mismatch")
    writer.write(FRAME_HEADER.pack(len(header_bytes), payload_length))
    writer.write(header_bytes)
    # ``directory`` is the first segment, but keep this explicit so the
    # lifetime of the segment list is obvious while the transport drains.
    for segment in segments:
        writer.write(segment)
    if drain:
        await writer.drain()


def encode_batch_blob(values: list[bytes | bytearray | memoryview | None]) -> bytearray:
    """Pack batch values into one bounded, zero-copy-decodable blob.

    Offsets are relative to the start of this blob.  Missing keys have a zero
    offset/length entry and do not consume payload space.  A bytearray is
    returned so the server can hand the single allocation directly to the
    asyncio transport.
    """
    _, segments, total_length = _batch_blob_layout(values)
    blob = bytearray(total_length)
    offset = 0
    for segment in segments:
        end = offset + len(segment)
        blob[offset:end] = segment
        offset = end
    return blob


def decode_batch_blob(
    blob: bytes | bytearray | memoryview,
    *,
    expected_count: int,
) -> list[memoryview | None]:
    """Validate and return views into a batch blob without copying objects."""
    view = memoryview(blob)
    entries = decode_batch_blob_directory(
        view,
        payload_length=len(view),
        expected_count=expected_count,
    )
    result: list[memoryview | None] = []
    for entry in entries:
        if entry is None:
            result.append(None)
            continue
        offset, length = entry
        result.append(view[offset : offset + length])
    return result


def decode_batch_blob_directory(
    directory: bytes | bytearray | memoryview,
    *,
    payload_length: int,
    expected_count: int,
) -> list[tuple[int, int] | None]:
    """Validate an MKVB directory before its object payloads are received.

    The normal decoder receives a single contiguous blob.  The streaming GET
    path only has the directory when it needs to decide how many bytes to read
    for the next object, so this helper deliberately validates offsets against
    the declared frame payload length instead of ``len(directory)``.
    """
    if expected_count < 0 or expected_count > MAX_BATCH_OBJECTS:
        raise ValueError("invalid MaKV batch object count")
    if payload_length < 0 or payload_length > DEFAULT_MAX_PAYLOAD_BYTES:
        raise ValueError("invalid MaKV batch blob payload length")
    view = memoryview(directory)
    if len(view) < BATCH_BLOB_HEADER.size:
        raise ValueError("truncated MaKV batch blob header")
    magic, version, count, total_length = BATCH_BLOB_HEADER.unpack_from(view, 0)
    if magic != BATCH_BLOB_MAGIC:
        raise ValueError("invalid MaKV batch blob magic")
    if version != BATCH_BLOB_VERSION:
        raise ValueError(f"unsupported MaKV batch blob version {version}")
    if count != expected_count:
        raise ValueError("MaKV batch blob object count mismatch")
    if total_length != payload_length:
        raise ValueError("MaKV batch blob length mismatch")
    directory_end = BATCH_BLOB_HEADER.size + count * BATCH_BLOB_ENTRY.size
    if directory_end > len(view) or directory_end > total_length:
        raise ValueError("MaKV batch blob directory is out of bounds")

    result: list[tuple[int, int] | None] = []
    ranges: list[tuple[int, int]] = []
    for index in range(count):
        found, offset, length = BATCH_BLOB_ENTRY.unpack_from(
            view, BATCH_BLOB_HEADER.size + index * BATCH_BLOB_ENTRY.size
        )
        if found not in (0, 1):
            raise ValueError("invalid MaKV batch blob found flag")
        if not found:
            if offset != 0 or length != 0:
                raise ValueError("missing MaKV batch entry has a payload range")
            result.append(None)
            continue
        if offset < directory_end or offset > total_length:
            raise ValueError("MaKV batch payload offset is out of bounds")
        if length > total_length - offset:
            raise ValueError("MaKV batch payload length is out of bounds")
        end = offset + length
        ranges.append((offset, end))
        result.append((offset, length))

    previous_end = directory_end
    for start, end in sorted(ranges):
        if start < previous_end:
            raise ValueError("MaKV batch payload ranges overlap")
        previous_end = max(previous_end, end)
    return result
