"""Token-level N-gram model for context mixing.

Maintains order-1 through order-N context tables with interpolated
backoff smoothing. Used as a fast, lightweight predictor alongside
the LLM in an ensemble. All operations are deterministic for
lossless codec symmetry.

Uses flat numpy arrays for inner storage instead of nested Python
dicts. This eliminates millions of small dict objects (~3.6 GB →
~1 GB per worker) and replaces the O(K) Python iteration loop in
predict() with a single numpy fancy-indexing call that runs at
C level, drastically reducing GIL hold time with 8+ threads.
"""

import numpy as np


def _context_hash(context_tokens, order):
    """Deterministic 64-bit hash of the last *order* tokens.

    Replaces tuple(context_tokens[-order:]) as dict key, eliminating
    per-token tuple allocations and reducing GC pressure.
    """
    h = 0
    end = len(context_tokens)
    for i in range(end - order, end):
        h = (h * 49157 + context_tokens[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class NgramModel:
    """Interpolated N-gram model operating on token IDs.

    Uses iterative interpolation: higher-order models are progressively
    blended with lower orders, weighted by context frequency. Unseen
    contexts fall back smoothly to lower orders down to unigram.

    The model updates online after each observed token, so it adapts
    to the specific document being compressed.

    Inner storage uses flat numpy arrays indexed by slot number.
    The outer dict (context_hash → slot) preserves insertion order
    for deterministic FIFO eviction.
    """

    # Smoothing constant for interpolation weights.
    ESCAPE = 5

    # Maximum context entries per order.
    MAX_TABLE_ENTRIES = 500_000

    # Maximum unique continuations per context.
    MAX_INNER_ENTRIES = 64

    def __init__(self, max_order: int = 4, vocab_size: int = 49152):
        self.max_order = max_order
        self.vocab_size = vocab_size

        # Order-0 (unigram) counts: dense array for fast vector ops.
        self.unigram_counts = np.zeros(vocab_size, dtype=np.float64)
        self.total_unigram = 0

        # Order 1..N: context_hash → slot_index.
        # Python dict preserves insertion order for FIFO eviction.
        self._slot_map: list = [None] + [dict() for _ in range(max_order)]

        # Flat inner storage per order.  Each context maps to a "slot"
        # containing up to MAX_INNER_ENTRIES (token_id, count) pairs.
        # Entries within a slot are kept in insertion order so that
        # argmin tie-breaking matches the old dict-based behavior.
        self._inner_ids: list = [None] + [
            np.empty((self.MAX_TABLE_ENTRIES, self.MAX_INNER_ENTRIES),
                     dtype=np.int32)
            for _ in range(max_order)
        ]
        self._inner_counts: list = [None] + [
            np.empty((self.MAX_TABLE_ENTRIES, self.MAX_INNER_ENTRIES),
                     dtype=np.int32)
            for _ in range(max_order)
        ]
        self._inner_sizes: list = [None] + [
            np.zeros(self.MAX_TABLE_ENTRIES, dtype=np.int16)
            for _ in range(max_order)
        ]
        self._ctx_totals: list = [None] + [
            np.zeros(self.MAX_TABLE_ENTRIES, dtype=np.int32)
            for _ in range(max_order)
        ]

        # Slot allocation: sequential counter + free list for recycling.
        self._next_slot = [0] * (max_order + 1)
        self._free_slots: list = [None] + [[] for _ in range(max_order)]

        # Pre-allocated buffers for building order predictions.
        self._buf = np.zeros(vocab_size, dtype=np.float64)
        self._probs = np.zeros(vocab_size, dtype=np.float64)

    def reset(self):
        """Reset all counts. Call when starting a new sequence."""
        self.unigram_counts[:] = 0
        self.total_unigram = 0
        self._slot_map = [None] + [dict() for _ in range(self.max_order)]
        self._next_slot = [0] * (self.max_order + 1)
        self._free_slots = [None] + [[] for _ in range(self.max_order)]
        for order in range(1, self.max_order + 1):
            self._inner_sizes[order][:] = 0
            self._ctx_totals[order][:] = 0

    def predict(self, context_tokens: list[int]) -> np.ndarray:
        """Predict next-token distribution given context.

        Uses numpy fancy indexing instead of Python dict iteration,
        replacing up to 256 Python loop iterations with C-level
        array operations that minimize GIL hold time.

        Args:
            context_tokens: List of preceding token IDs.

        Returns:
            numpy array of shape (vocab_size,) with probabilities summing to ~1.
        """
        # Start with unigram (Laplace-smoothed)
        probs = self._probs
        np.add(self.unigram_counts, 1.0, out=probs)
        probs /= (self.total_unigram + self.vocab_size)

        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break

            ctx = _context_hash(context_tokens, order)
            slot = self._slot_map[order].get(ctx)
            if slot is None:
                continue

            total = int(self._ctx_totals[order][slot])
            if total == 0:
                continue

            lam = total / (total + self.ESCAPE)

            # Vectorized inner loop: single numpy fancy-index call
            # replaces K Python dict iterations (K up to 64).
            buf = self._buf
            buf[:] = 0
            size = int(self._inner_sizes[order][slot])
            ids = self._inner_ids[order][slot, :size]
            cts = self._inner_counts[order][slot, :size]
            buf[ids] = cts   # C-level scatter — the key optimization
            buf /= buf.sum()

            # Blend: probs = lam * order_k + (1-lam) * probs
            probs *= (1.0 - lam)
            buf *= lam
            probs += buf

        return probs

    def _alloc_slot(self, order: int) -> int:
        """Get a free slot index, recycling evicted slots first."""
        if self._free_slots[order]:
            return self._free_slots[order].pop()
        slot = self._next_slot[order]
        self._next_slot[order] += 1
        return slot

    def update(self, context_tokens: list[int], actual_token: int):
        """Update counts after observing a token.

        Must be called identically during compression and decompression
        to maintain codec symmetry.

        Args:
            context_tokens: Context that preceded the token.
            actual_token: The token that was actually observed.
        """
        # Update unigram
        self.unigram_counts[actual_token] += 1
        self.total_unigram += 1

        # Update higher orders
        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break

            ctx = _context_hash(context_tokens, order)
            slot_map = self._slot_map[order]

            # Evict oldest context if table is full and this is new
            if ctx not in slot_map and len(slot_map) >= self.MAX_TABLE_ENTRIES:
                evict_ctx = next(iter(slot_map))
                evict_slot = slot_map.pop(evict_ctx)
                self._free_slots[order].append(evict_slot)

            if ctx in slot_map:
                slot = slot_map[ctx]
                size = int(self._inner_sizes[order][slot])
                ids = self._inner_ids[order][slot]
                counts = self._inner_counts[order][slot]

                # Search for actual_token (numpy vectorized)
                mask = ids[:size] == actual_token
                if mask.any():
                    # Token exists: increment its count
                    idx = int(np.argmax(mask))
                    counts[idx] += 1
                    self._ctx_totals[order][slot] += 1
                elif size < self.MAX_INNER_ENTRIES:
                    # New token, space available: append
                    ids[size] = actual_token
                    counts[size] = 1
                    self._inner_sizes[order][slot] = size + 1
                    self._ctx_totals[order][slot] += 1
                else:
                    # Full (64 entries). Simulate the original add-then-evict:
                    # new entry has count=1, evicted entry has count ≤ 1 = 1,
                    # so net total change is always 0.
                    min_count = int(counts[:size].min())
                    if min_count == 1:
                        # Evict oldest entry with count=1, add new at end.
                        # Shift maintains insertion order so argmin
                        # tie-breaking matches the original dict behavior.
                        min_idx = int(np.argmin(counts[:size]))
                        if min_idx < size - 1:
                            ids[min_idx:size-1] = ids[min_idx+1:size]
                            counts[min_idx:size-1] = counts[min_idx+1:size]
                        ids[size - 1] = actual_token
                        counts[size - 1] = 1
                    # else: min_count > 1, new entry would be sole minimum
                    # and immediately evicted — no-op on entries and total.
            else:
                # New context: allocate a slot
                slot = self._alloc_slot(order)
                slot_map[ctx] = slot
                self._inner_ids[order][slot, 0] = actual_token
                self._inner_counts[order][slot, 0] = 1
                self._inner_sizes[order][slot] = 1
                self._ctx_totals[order][slot] = 1
