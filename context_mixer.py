"""Adaptive context mixer for model blending.

Combines multiple probability distributions using linear mixing
with online multiplicative weight updates. The key idea:
models that predict well get higher weights, adapted after every
token.

All operations are deterministic for lossless codec symmetry.

Uses numpy instead of torch for CPU tensor operations to minimize
per-operation dispatch overhead.
"""

import math
import numpy as np


class ContextMixer:
    """Linear context mixer with online weight adaptation.

    Linear mixing computes a weighted average of input distributions:

        mixed[i] = sum(w_j * p_j[i])

    This preserves the dominant model's confidence — if the LLM puts
    0.90 on a token and has weight 0.85, it contributes 0.765 to the
    mix. Unlike geometric mixing, near-uniform secondary models don't
    flatten the distribution.

    Weights are updated using the exponential weights algorithm
    (multiplicative updates based on per-model log-loss). Models that
    predict accurately gain weight automatically.
    """

    # Floor for probabilities before taking log, to avoid -inf.
    PROB_FLOOR = 1e-8

    def __init__(self, num_models: int, lr: float = 0.5,
                 initial_weights: list[float] = None,
                 vocab_size: int = 49152):
        """Initialize the mixer.

        Args:
            num_models: Number of models to mix.
            lr: Learning rate for weight adaptation. Higher values
                make the mixer react faster to model performance.
                0 = static equal weights; 1 = full Bayesian updating.
            initial_weights: Starting weights for each model. If None,
                uses LLM-dominant defaults: first model gets 0.85,
                rest share 0.15 equally. This prevents uninformed
                secondary models from diluting the LLM early on.
            vocab_size: Vocabulary size for pre-allocating mix buffers.
        """
        self.num_models = num_models
        self.lr = lr

        if initial_weights is not None:
            assert len(initial_weights) == num_models
            self._initial_weights = list(initial_weights)
        else:
            # LLM-dominant: first model (LLM) gets 0.85,
            # remaining 0.15 split equally among secondary models.
            if num_models == 1:
                self._initial_weights = [1.0]
            else:
                secondary_w = 0.15 / (num_models - 1)
                self._initial_weights = (
                    [0.85] + [secondary_w] * (num_models - 1)
                )

        self._init_from_weights(self._initial_weights)

        # Pre-allocated buffers for zero-alloc mixing.
        self._mix_buf = np.zeros(vocab_size, dtype=np.float64)
        self._scale_buf = np.zeros(vocab_size, dtype=np.float64)

    def _init_from_weights(self, weights: list[float]):
        """Set log_weights and weights from a normalized weight list."""
        total = sum(weights)
        self.weights = [w / total for w in weights]
        # Keep log-space copies for numerically stable updates.
        self.log_weights = [math.log(w + 1e-30) for w in self.weights]

    def reset(self):
        """Reset to initial weights. Call when starting a new sequence."""
        self._init_from_weights(self._initial_weights)

    def mix(self, prob_list: list[np.ndarray]) -> np.ndarray:
        """Combine multiple probability distributions.

        Uses linear mixing:
            mixed[i] = sum(w_j * p_j[i])

        Args:
            prob_list: List of numpy arrays, each shape (vocab_size,),
                       each summing to ~1.

        Returns:
            numpy array of shape (vocab_size,) with blended probabilities.
        """
        if len(prob_list) != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models, got {len(prob_list)}"
            )

        if self.num_models == 1:
            return prob_list[0]

        # Weighted linear combination (in-place, zero-alloc).
        # Avoids w * probs temporary (384 KB per model per token).
        mixed = self._mix_buf
        mixed[:] = 0
        scale_buf = self._scale_buf
        for w, probs in zip(self.weights, prob_list):
            np.multiply(probs, w, out=scale_buf)
            mixed += scale_buf

        return mixed

    def update(self, actual_token: int, prob_list: list[np.ndarray]):
        """Update weights based on observed token.

        Uses the exponential weights algorithm: each model's weight is
        multiplied by P(actual_token | model)^lr, then renormalized.
        Models that predicted the actual token well gain weight.

        Must be called identically during compression and decompression.

        Args:
            actual_token: The token that was actually observed.
            prob_list: Same list passed to mix() for this token.
        """
        if self.num_models <= 1:
            return

        for i, probs in enumerate(prob_list):
            p = max(float(probs[actual_token]), self.PROB_FLOOR)
            self.log_weights[i] += self.lr * math.log(p)

        # Normalize weights (subtract max for numerical stability)
        max_lw = max(self.log_weights)
        raw = [math.exp(lw - max_lw) for lw in self.log_weights]
        total = sum(raw)
        self.weights = [w / total for w in raw]

    def get_weights(self) -> list[float]:
        """Return current mixer weights (for diagnostics)."""
        return list(self.weights)
