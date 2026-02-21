"""Online adaptive bias layer for per-document learning.

Applies a learned bias vector in log-probability space on top of the
LLM's output. The bias is updated via gradient descent after each
observed token, allowing the model to adapt to the specific document
being compressed.

Both compressor and decompressor start from the same initial state
(zero bias) and apply identical updates, maintaining lossless symmetry.

Uses float64 numpy throughout to ensure bit-exact reproducibility
across compression and decompression.
"""

import numpy as np


class AdaptiveHead:
    """Thin adaptive bias layer on top of LLM probabilities.

    Instead of fine-tuning the full model, this learns a per-token
    bias correction:

        adjusted[i] = softmax(log(probs[i]) + bias[i])

    This is equivalent to multiplicatively rescaling each probability:

        adjusted[i] = probs[i] * exp(bias[i]) / Z

    The bias starts at zero (identity transform) and is updated after
    each observed token using the gradient of cross-entropy loss. Over
    time, it learns to boost tokens that the LLM under-predicts for
    this specific document and suppress over-predicted ones.
    """

    def __init__(self, vocab_size: int = 49152, lr: float = 0.001):
        """Initialize the adaptive head.

        Args:
            vocab_size: Size of the token vocabulary.
            lr: Learning rate for bias updates. Small values (0.001)
                give gentle adaptation; larger values risk oscillation.
        """
        self.vocab_size = vocab_size
        self.lr = lr

        # Bias in log-probability space. float64 for precision.
        self.bias = np.zeros(vocab_size, dtype=np.float64)

        # Pre-allocated buffers to avoid per-token allocation.
        self._log_buf = np.zeros(vocab_size, dtype=np.float64)
        self._grad_buf = np.zeros(vocab_size, dtype=np.float64)

    def reset(self):
        """Reset bias to zero. Call when starting a new sequence."""
        self.bias[:] = 0

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        """Apply adaptive bias to LLM probabilities.

        Args:
            probs: numpy array of shape (vocab_size,) from the LLM,
                   summing to ~1.

        Returns:
            Adjusted probabilities, float64 numpy array, summing to ~1.
        """
        # Work in float64 for precision
        log_buf = self._log_buf
        np.log(probs + 1e-10, out=log_buf)
        log_buf += self.bias

        # Numerically stable softmax
        log_buf -= log_buf.max()
        np.exp(log_buf, out=log_buf)
        log_buf /= log_buf.sum()

        return log_buf

    def update(self, actual_token: int, adjusted_probs: np.ndarray):
        """Update bias after observing a token.

        Performs one step of gradient descent on cross-entropy loss:
            L = -log(adjusted_probs[actual_token])
            dL/d(bias[i]) = adjusted_probs[i] - 1_{i == actual_token}

        Must be called identically during compression and decompression.

        Args:
            actual_token: The token that was actually observed.
            adjusted_probs: The probabilities returned by adjust() for
                            this token (before mixing with other models).
        """
        grad = self._grad_buf
        np.copyto(grad, adjusted_probs)
        grad[actual_token] -= 1.0
        self.bias -= self.lr * grad
