"""Utility functions for the neural compressor."""

import numpy as np


# CDF precision — probabilities are quantized to integers summing to this value.
# Must be a power of 2 and fit comfortably in the arithmetic coder's range.
CDF_TOTAL = 1 << 16  # 65536

# Minimum probability assigned to any symbol to avoid zero-width intervals.
MIN_PROB = 1


def probs_to_cdf(probs: np.ndarray, total: int = CDF_TOTAL) -> np.ndarray:
    """Convert a probability distribution to an integer CDF for arithmetic coding.

    Ensures every symbol gets at least MIN_PROB counts so the arithmetic
    coder never encounters a zero-width interval.

    Uses numpy operations instead of torch for lower dispatch overhead.

    Args:
        probs: numpy array of shape (vocab_size,) with probabilities.
        total: CDF total (sum of all counts).

    Returns:
        numpy array of shape (vocab_size + 1,) with cdf[0] = 0, cdf[-1] = total.
    """
    n = probs.shape[0]

    # Scale probabilities to integer counts
    counts = (probs * (total - n * MIN_PROB)).astype(np.int64).clip(min=0) + MIN_PROB

    # Adjust to hit exact total (distribute rounding error)
    diff = total - counts.sum()
    if diff != 0:
        counts[counts.argmax()] += diff

    # Build CDF via vectorized cumsum
    cdf = np.empty(n + 1, dtype=np.int64)
    cdf[0] = 0
    np.cumsum(counts, out=cdf[1:])

    return cdf


class CdfConverter:
    """Zero-allocation CDF converter with pre-allocated buffers.

    Replaces per-token calls to probs_to_cdf(), eliminating ~1.9 MB of
    temporary numpy allocations per token (5 × 384 KB arrays).

    The returned CDF array is an internal buffer — callers must consume
    it before the next convert() call.
    """

    __slots__ = ('_n', '_float_buf', '_counts', '_cdf')

    def __init__(self, vocab_size: int):
        self._n = vocab_size
        self._float_buf = np.zeros(vocab_size, dtype=np.float64)
        self._counts = np.zeros(vocab_size, dtype=np.int64)
        self._cdf = np.zeros(vocab_size + 1, dtype=np.int64)

    def convert(self, probs: np.ndarray, total: int = CDF_TOTAL) -> np.ndarray:
        """Convert probabilities to CDF without allocations.

        Produces identical output to probs_to_cdf().
        """
        n = self._n
        scale = total - n * MIN_PROB

        # probs * scale → float buffer (in-place)
        np.multiply(probs, scale, out=self._float_buf)

        # Truncate to int64 (same as .astype(np.int64))
        self._counts[:] = self._float_buf

        # clip(min=0) + MIN_PROB (in-place)
        np.clip(self._counts, 0, None, out=self._counts)
        self._counts += MIN_PROB

        # Adjust to hit exact total
        diff = total - self._counts.sum()
        if diff != 0:
            self._counts[self._counts.argmax()] += diff

        # Build CDF via cumsum (in-place)
        self._cdf[0] = 0
        np.cumsum(self._counts, out=self._cdf[1:])

        return self._cdf


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
