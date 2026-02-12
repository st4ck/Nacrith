"""Tests for the full neural compressor pipeline — slow (need model)."""

import sys
import os
import gzip
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.slow
def test_roundtrip_hello(compressor):
    text = "Hello, World!"
    assert compressor.decompress(compressor.compress(text)) == text


@pytest.mark.slow
def test_roundtrip_paragraph(compressor, sample_texts):
    text = sample_texts["medium"]
    assert compressor.decompress(compressor.compress(text)) == text


@pytest.mark.slow
def test_roundtrip_unicode(compressor):
    text = "café résumé naïve"
    assert compressor.decompress(compressor.compress(text)) == text


@pytest.mark.slow
def test_roundtrip_empty(compressor):
    assert compressor.decompress(compressor.compress("")) == ""


@pytest.mark.slow
def test_roundtrip_single_char(compressor):
    text = "A"
    assert compressor.decompress(compressor.compress(text)) == text


@pytest.mark.slow
def test_roundtrip_repeated(compressor):
    text = "a" * 500
    assert compressor.decompress(compressor.compress(text)) == text


@pytest.mark.slow
def test_compression_ratio(compressor, sample_texts):
    text = sample_texts["long"]
    compressed = compressor.compress(text)
    original_size = len(text.encode("utf-8"))
    compressed_size = len(compressed)
    assert compressed_size < original_size, (
        f"Compressed ({compressed_size}) should be < original ({original_size})"
    )


@pytest.mark.slow
def test_vs_gzip(compressor, sample_texts):
    text = sample_texts["long"]
    neural = compressor.compress(text)
    gz = gzip.compress(text.encode("utf-8"))
    original = len(text.encode("utf-8"))
    print(f"\nOriginal: {original}B  Neural: {len(neural)}B  gzip: {len(gz)}B")


@pytest.mark.slow
def test_magic_bytes(compressor):
    compressed = compressor.compress("Test magic bytes")
    assert compressed[:4] == b"NC01"


@pytest.mark.slow
def test_roundtrip_numbers(compressor):
    text = "The year 2024 had 365 days and 12 months."
    assert compressor.decompress(compressor.compress(text)) == text
