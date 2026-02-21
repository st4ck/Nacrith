"""Lempel-Ziv Prediction (LZP) model for context mixing.

Finds the longest matching context in the token history and predicts
the token that followed that match. Effective for repetitive or
structured text (code, CSV, legal documents, etc.) where exact
sub-sequences recur.

All operations are deterministic for lossless codec symmetry.

Uses numpy instead of torch for CPU tensor operations to minimize
per-operation dispatch overhead.
"""

import numpy as np


def _context_hash(context_tokens, order):
    """Deterministic 64-bit hash of the last *order* tokens.

    Seeds with *order* so different-length contexts that share a
    suffix hash differently (all orders share a single table).
    """
    h = order
    end = len(context_tokens)
    for i in range(end - order, end):
        h = (h * 49157 + context_tokens[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class LZPModel:
    """LZP match model operating on token IDs.

    For each context length from min_order to max_order, stores the
    most recently observed next-token. At prediction time, uses the
    highest-order match available, assigning high probability to
    the predicted token.

    This complements the N-gram model: N-gram captures short local
    statistics, while LZP captures exact long-range repetitions.

    Table is capped at MAX_TABLE_ENTRIES to bound memory. Oldest
    entries are evicted first (FIFO), deterministic for codec symmetry.
    """

    # Probability mass assigned to the matched token.
    MATCH_PROB = 0.85

    # Maximum entries in the combined table (all orders share one dict).
    MAX_TABLE_ENTRIES = 1_000_000

    def __init__(self, max_order: int = 8, min_order: int = 4,
                 vocab_size: int = 49152):
        self.max_order = max_order
        self.min_order = min_order
        self.vocab_size = vocab_size

        # Maps context_hash -> last seen next token (all orders share one dict).
        # We store only the most recent next-token (classic LZP approach).
        self._tables: dict = {}

        # Pre-allocated buffers (avoid per-token allocation)
        self._residual = (1.0 - self.MATCH_PROB) / (vocab_size - 1)
        self._match_buf = np.full(vocab_size, self._residual, dtype=np.float64)
        self._uniform = np.full(
            vocab_size, 1.0 / vocab_size, dtype=np.float64,
        )

    def reset(self):
        """Reset all match tables. Call when starting a new sequence."""
        self._tables.clear()

    def predict(self, context_tokens: list[int]) -> np.ndarray:
        """Predict next-token distribution based on longest context match.

        Searches from highest order down to min_order. Returns a peaked
        distribution if a match is found, or uniform if no match exists.

        Args:
            context_tokens: List of preceding token IDs.

        Returns:
            numpy array of shape (vocab_size,) with probabilities summing to 1.
        """
        # Try from highest to lowest order
        for order in range(self.max_order, self.min_order - 1, -1):
            if len(context_tokens) < order:
                continue

            ctx = _context_hash(context_tokens, order)
            predicted = self._tables.get(ctx)

            if predicted is not None:
                # Found a match: peaked distribution on predicted token.
                # Use pre-allocated buffer, reset to residual, set peak.
                buf = self._match_buf
                buf[:] = self._residual
                buf[predicted] = self.MATCH_PROB
                return buf

        # No match at any order: return pre-allocated uniform distribution.
        return self._uniform

    def update(self, context_tokens: list[int], actual_token: int):
        """Record the next token for each context length.

        Must be called identically during compression and decompression
        to maintain codec symmetry.

        Args:
            context_tokens: Context that preceded the token.
            actual_token: The token that was actually observed.
        """
        for order in range(self.min_order, self.max_order + 1):
            if len(context_tokens) < order:
                break
            key = _context_hash(context_tokens, order)

            # Evict oldest entry if table is full and this is a new key
            if key not in self._tables and len(self._tables) >= self.MAX_TABLE_ENTRIES:
                evict_key = next(iter(self._tables))
                del self._tables[evict_key]

            self._tables[key] = actual_token
