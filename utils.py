"""Utility functions for the neural compressor."""

import torch


# CDF precision — probabilities are quantized to integers summing to this value.
# Must be a power of 2 and fit comfortably in the arithmetic coder's range.
CDF_TOTAL = 1 << 16  # 65536

# Minimum probability assigned to any symbol to avoid zero-width intervals.
MIN_PROB = 1


def probs_to_cdf(probs: torch.Tensor, total: int = CDF_TOTAL) -> list[int]:
    """Convert a probability distribution to an integer CDF for arithmetic coding.

    Ensures every symbol gets at least MIN_PROB counts so the arithmetic
    coder never encounters a zero-width interval.

    Args:
        probs: Tensor of shape (vocab_size,) with probabilities.
        total: CDF total (sum of all counts).

    Returns:
        List of length vocab_size + 1, where cdf[0] = 0, cdf[-1] = total.
    """
    n = len(probs)

    # Scale probabilities to integer counts
    counts = (probs * (total - n * MIN_PROB)).long()
    counts = counts.clamp(min=0)

    # Ensure minimum probability for every symbol
    counts = counts + MIN_PROB

    # Adjust to hit exact total (distribute rounding error)
    diff = total - counts.sum().item()
    if diff != 0:
        # Add/subtract from the largest count
        max_idx = counts.argmax().item()
        counts[max_idx] += diff

    # Build CDF
    cdf = [0]
    running = 0
    for c in counts.tolist():
        running += int(c)
        cdf.append(running)

    # Safety: ensure last entry is exactly total
    cdf[-1] = total

    return cdf


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
