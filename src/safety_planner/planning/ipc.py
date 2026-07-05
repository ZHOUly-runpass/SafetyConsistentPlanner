from __future__ import annotations

import struct
from typing import BinaryIO, Callable

import msgpack
import numpy as np


MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def write_message(stream: BinaryIO, payload: dict) -> None:
    encoded = msgpack.packb(_encode(payload), use_bin_type=True)
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds the configured size limit.")
    stream.write(struct.pack(">I", len(encoded)))
    stream.write(encoded)
    stream.flush()


def read_message(stream: BinaryIO) -> dict | None:
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise EOFError("Truncated IPC message header.")
    size = struct.unpack(">I", header)[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds the configured size limit.")
    body = stream.read(size)
    if len(body) != size:
        raise EOFError("Truncated IPC message body.")
    payload = _decode(msgpack.unpackb(body, raw=False))
    if not isinstance(payload, dict):
        raise ValueError("IPC root payload must be a mapping.")
    return payload


def serve_messages(
    reader: BinaryIO,
    writer: BinaryIO,
    handler: Callable[[dict], dict],
) -> None:
    while True:
        request = read_message(reader)
        if request is None:
            return
        request_id = request.get("request_id")
        try:
            if request.get("schema_version") != "1.0":
                raise ValueError("Unsupported IPC schema_version.")
            response = handler(request)
            response = {"schema_version": "1.0", "request_id": request_id, **response}
        except Exception as exc:
            response = {
                "schema_version": "1.0",
                "request_id": request_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        write_message(writer, response)


def _encode(value):
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": contiguous.shape,
            "data": contiguous.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"])).copy()
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value

