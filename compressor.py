"""
Neural text compressor using SmolLM2-135M + arithmetic coding.

Lossless compression: the compressor and decompressor share the exact
same model and produce identical probability distributions, ensuring
perfect reconstruction.
"""

import struct
import sys

from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder
from model_wrapper import ModelWrapper
from utils import probs_to_cdf, CDF_TOTAL

# File format:
# [4 bytes] Magic: b'NC01'
# [4 bytes] Number of tokens (uint32, big-endian)
# [4 bytes] Number of compressed bits (uint32, big-endian)
# [rest]    Arithmetic-coded bitstream

MAGIC = b'NC01'
HEADER_SIZE = 12  # 4 + 4 + 4


class NeuralCompressor:
    """Lossless neural text compressor."""

    def __init__(self, model: ModelWrapper = None, verbose: bool = True):
        self.verbose = verbose
        self.model = model or ModelWrapper(verbose=verbose)

    def compress(self, text: str) -> bytes:
        """Compress text to bytes.

        Args:
            text: Input text string.

        Returns:
            Compressed data as bytes (header + arithmetic-coded stream).
        """
        if not text:
            # Empty text: just header with 0 tokens
            return MAGIC + struct.pack('>II', 0, 0)

        # Tokenize
        token_ids = self.model.tokenizer.encode(text)
        num_tokens = len(token_ids)

        if self.verbose:
            print(f"Tokens: {num_tokens}", file=sys.stderr)

        # Encode with arithmetic coder
        encoder = ArithmeticEncoder()
        context = []
        self.model.reset_cache()

        for i, token_id in enumerate(token_ids):
            if self.verbose and (i % 100 == 0 or i == num_tokens - 1):
                print(
                    f"\rCompressing: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)",
                    end="", file=sys.stderr,
                )

            # Get model prediction
            probs = self.model.get_probs(context)
            cdf = probs_to_cdf(probs, CDF_TOTAL)

            # Encode the actual token
            encoder.encode_symbol(cdf, token_id)

            # Update context (chunked sliding window — must match model_wrapper)
            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK
                context = context[-keep:]

        if self.verbose:
            print(file=sys.stderr)

        # Finalize
        compressed_bits = encoder.get_bit_count()
        stream = encoder.finish()

        # Build output: header + stream
        header = MAGIC + struct.pack('>II', num_tokens, compressed_bits)
        return header + stream

    def decompress(self, data: bytes) -> str:
        """Decompress bytes back to text.

        Args:
            data: Compressed data (as produced by compress()).

        Returns:
            Original text string.
        """
        if len(data) < HEADER_SIZE:
            raise ValueError("Data too short to contain header")

        # Parse header
        magic = data[:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r} (expected {MAGIC!r})")

        num_tokens, compressed_bits = struct.unpack('>II', data[4:HEADER_SIZE])

        if num_tokens == 0:
            return ""

        # Decode with arithmetic coder
        stream = data[HEADER_SIZE:]
        decoder = ArithmeticDecoder(stream)
        context = []
        token_ids = []
        self.model.reset_cache()

        for i in range(num_tokens):
            if self.verbose and (i % 100 == 0 or i == num_tokens - 1):
                print(
                    f"\rDecompressing: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)",
                    end="", file=sys.stderr,
                )

            # Get model prediction (same as compressor)
            probs = self.model.get_probs(context)
            cdf = probs_to_cdf(probs, CDF_TOTAL)

            # Decode the token
            token_id = decoder.decode_symbol(cdf)
            token_ids.append(token_id)

            # Update context (chunked sliding window — must match compressor)
            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK
                context = context[-keep:]

        if self.verbose:
            print(file=sys.stderr)

        # Detokenize
        text = self.model.tokenizer.decode(token_ids)
        return text
