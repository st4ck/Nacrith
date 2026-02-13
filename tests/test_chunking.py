"""Tests for _segment_chunks() — fast, no model needed."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compressor import (
    _segment_chunks,
    CHUNK_TYPE_TEXT,
    CHUNK_TYPE_BINARY,
    MIN_TEXT_RUN,
    MAX_BRIDGE_GAP,
    MIN_BINARY_CHUNK,
)


def test_segment_empty():
    assert _segment_chunks(b"") == []


def test_segment_all_text():
    data = b"Hello, this is a perfectly normal ASCII text string that is long enough to qualify." * 2
    chunks = _segment_chunks(data)
    assert len(chunks) == 1
    ctype, offset, length = chunks[0]
    assert ctype == CHUNK_TYPE_TEXT
    assert offset == 0
    assert length == len(data)


def test_segment_all_binary():
    data = bytes(range(128, 256)) * 4
    chunks = _segment_chunks(data)
    assert len(chunks) == 1
    ctype, offset, length = chunks[0]
    assert ctype == CHUNK_TYPE_BINARY
    assert offset == 0
    assert length == len(data)


def test_segment_short_text_becomes_binary():
    """Text runs shorter than MIN_TEXT_RUN get demoted to binary."""
    short_text = b"Hi"  # Way shorter than MIN_TEXT_RUN
    binary = bytes([0x80] * 100)
    data = binary + short_text + binary
    chunks = _segment_chunks(data)
    # Short text should be merged into binary
    assert len(chunks) == 1
    assert chunks[0][0] == CHUNK_TYPE_BINARY
    assert chunks[0][2] == len(data)


def test_segment_bridge_small_gap():
    """Small binary gap between two text runs gets bridged."""
    text = b"A" * MIN_TEXT_RUN
    gap = bytes([0x80] * MAX_BRIDGE_GAP)  # Exactly at threshold
    data = text + gap + text
    chunks = _segment_chunks(data)
    assert len(chunks) == 1
    assert chunks[0][0] == CHUNK_TYPE_TEXT
    assert chunks[0][2] == len(data)


def test_segment_no_bridge_large_gap():
    """Binary gap >= MIN_BINARY_CHUNK stays as its own chunk."""
    text = b"A" * MIN_TEXT_RUN
    gap = bytes([0x80] * MIN_BINARY_CHUNK)
    data = text + gap + text
    chunks = _segment_chunks(data)
    assert len(chunks) == 3
    assert chunks[0][0] == CHUNK_TYPE_TEXT
    assert chunks[1][0] == CHUNK_TYPE_BINARY
    assert chunks[2][0] == CHUNK_TYPE_TEXT


def test_segment_absorb_small_binary_into_text():
    """Small binary chunk between text runs gets absorbed into text."""
    text = b"A" * MIN_TEXT_RUN
    # Too big for bridging, but small enough for absorption
    gap = bytes([0x80] * (MAX_BRIDGE_GAP + 1))
    assert MAX_BRIDGE_GAP + 1 < MIN_BINARY_CHUNK  # precondition
    data = text + gap + text
    chunks = _segment_chunks(data)
    assert len(chunks) == 1
    assert chunks[0][0] == CHUNK_TYPE_TEXT
    assert chunks[0][2] == len(data)


def test_segment_covers_all_bytes():
    """Chunks must cover every byte of input exactly once."""
    # Mix of text and binary
    data = b"A" * 100 + bytes([0x80] * 50) + b"B" * 100 + bytes([0xFF] * 30)
    chunks = _segment_chunks(data)
    total = sum(length for _, _, length in chunks)
    assert total == len(data)


def test_segment_offsets_contiguous():
    """Chunk offsets must be contiguous with no gaps or overlaps."""
    data = b"Hello World! " * 20 + bytes(range(128, 200)) + b"More text here. " * 10
    chunks = _segment_chunks(data)
    expected_offset = 0
    for _, offset, length in chunks:
        assert offset == expected_offset, f"Gap or overlap at offset {offset}"
        expected_offset += length
    assert expected_offset == len(data)
