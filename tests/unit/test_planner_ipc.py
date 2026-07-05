from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from src.safety_planner.planning.ipc import read_message, serve_messages, write_message


def test_message_round_trip_preserves_numpy_arrays() -> None:
    stream = BytesIO()
    payload = {
        "schema_version": "1.0",
        "request_id": "abc",
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
    }
    write_message(stream, payload)
    stream.seek(0)
    decoded = read_message(stream)
    assert decoded["request_id"] == "abc"
    np.testing.assert_array_equal(decoded["array"], payload["array"])


def test_message_rejects_truncated_body() -> None:
    stream = BytesIO(b"\x00\x00\x00\x04x")
    with pytest.raises(EOFError, match="body"):
        read_message(stream)


def test_server_returns_versioned_error_without_crashing() -> None:
    reader = BytesIO()
    write_message(reader, {"schema_version": "0.0", "request_id": "bad"})
    reader.seek(0)
    writer = BytesIO()
    serve_messages(reader, writer, lambda request: {"ok": True})
    writer.seek(0)
    response = read_message(writer)
    assert response["request_id"] == "bad"
    assert response["ok"] is False
    assert response["error_type"] == "ValueError"
