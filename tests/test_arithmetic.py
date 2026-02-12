"""Tests for arithmetic coder — no model needed."""

import sys
import os
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder


def make_uniform_cdf(num_symbols, total=1 << 16):
    """Create a uniform CDF for num_symbols with given total."""
    step = total // num_symbols
    cdf = [i * step for i in range(num_symbols)]
    cdf.append(total)
    return cdf


def roundtrip(symbols, cdf):
    """Encode then decode, return decoded symbols."""
    encoder = ArithmeticEncoder()
    for s in symbols:
        encoder.encode_symbol(cdf, s)
    data = encoder.finish()

    decoder = ArithmeticDecoder(data)
    decoded = [decoder.decode_symbol(cdf) for _ in range(len(symbols))]
    return decoded


def test_roundtrip_basic():
    symbols = [0, 1, 2, 1, 0]
    cdf = make_uniform_cdf(3)
    assert roundtrip(symbols, cdf) == symbols


def test_roundtrip_uniform():
    random.seed(42)
    symbols = [random.randint(0, 9) for _ in range(100)]
    cdf = make_uniform_cdf(10)
    assert roundtrip(symbols, cdf) == symbols


def test_roundtrip_skewed():
    random.seed(42)
    total = 1 << 16
    # Symbol 0 gets 99% of probability mass
    high = int(total * 0.99)
    remaining = total - high
    n_other = 9
    step = remaining // n_other
    cdf = [0, high]
    for i in range(1, n_other):
        cdf.append(high + i * step)
    cdf.append(total)

    symbols = [0 if random.random() < 0.99 else random.randint(1, 9)
               for _ in range(100)]
    assert roundtrip(symbols, cdf) == symbols


def test_roundtrip_binary():
    random.seed(42)
    symbols = [random.randint(0, 1) for _ in range(100)]
    cdf = make_uniform_cdf(2)
    assert roundtrip(symbols, cdf) == symbols


def test_roundtrip_long_sequence():
    random.seed(42)
    symbols = [random.randint(0, 9) for _ in range(1500)]
    cdf = make_uniform_cdf(10)
    assert roundtrip(symbols, cdf) == symbols


def test_roundtrip_large_alphabet():
    random.seed(42)
    symbols = [random.randint(0, 999) for _ in range(100)]
    cdf = make_uniform_cdf(1000)
    assert roundtrip(symbols, cdf) == symbols


def test_single_symbol():
    symbols = [5]
    cdf = make_uniform_cdf(10)
    assert roundtrip(symbols, cdf) == symbols


def test_deterministic():
    symbols = [1, 2, 3, 4, 5, 1, 2, 3]
    cdf = make_uniform_cdf(10)

    encoder1 = ArithmeticEncoder()
    for s in symbols:
        encoder1.encode_symbol(cdf, s)
    data1 = encoder1.finish()

    encoder2 = ArithmeticEncoder()
    for s in symbols:
        encoder2.encode_symbol(cdf, s)
    data2 = encoder2.finish()

    assert data1 == data2


@pytest.mark.parametrize("length", [1, 5, 50, 500])
def test_parametrized_lengths(length):
    random.seed(42)
    symbols = [random.randint(0, 9) for _ in range(length)]
    cdf = make_uniform_cdf(10)
    assert roundtrip(symbols, cdf) == symbols


def test_varying_cdfs():
    """Each symbol encoded with a different CDF (simulates model predictions)."""
    random.seed(42)
    n = 50
    num_symbols = 5
    total = 1 << 16

    symbols = [random.randint(0, num_symbols - 1) for _ in range(n)]
    # Generate a different random CDF for each position
    cdfs = []
    for _ in range(n):
        weights = [random.randint(1, 100) for _ in range(num_symbols)]
        s = sum(weights)
        cdf = [0]
        running = 0
        for w in weights:
            running += (w * total) // s
            cdf.append(running)
        cdf[-1] = total
        cdfs.append(cdf)

    encoder = ArithmeticEncoder()
    for sym, cdf in zip(symbols, cdfs):
        encoder.encode_symbol(cdf, sym)
    data = encoder.finish()

    decoder = ArithmeticDecoder(data)
    decoded = []
    for cdf in cdfs:
        decoded.append(decoder.decode_symbol(cdf))

    assert decoded == symbols
