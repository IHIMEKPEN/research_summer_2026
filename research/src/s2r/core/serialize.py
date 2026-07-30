"""Fast msgpack serialization helpers."""

from __future__ import annotations

from typing import Any

import msgpack


def pack(obj: dict[str, Any]) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack(buf: bytes) -> dict[str, Any]:
    data = msgpack.unpackb(buf, raw=False)
    if not isinstance(data, dict):
        raise TypeError("Expected dict payload on wire")
    return data
