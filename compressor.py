"""
Neural text compressor using SmolLM2-135M + arithmetic coding.

Lossless compression: the compressor and decompressor share the exact
same model and produce identical probability distributions, ensuring
perfect reconstruction.
"""

import gzip
import lzma
import struct
import sys

from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder
from model_wrapper import ModelWrapper
from utils import probs_to_cdf, CDF_TOTAL

# NC01 text format:
# [4 bytes] Magic: b'NC01'
# [4 bytes] Number of tokens (uint32, big-endian)
# [4 bytes] Number of compressed bits (uint32, big-endian)
# [rest]    Arithmetic-coded bitstream

MAGIC = b'NC01'
MAGIC_BIN = b'NC02'
HEADER_SIZE = 12  # 4 + 4 + 4

# NC02 hybrid binary format constants
NC02_VERSION = 2
CHUNK_TYPE_TEXT = 0x54    # 'T'
CHUNK_TYPE_BINARY = 0x42  # 'B'
MIN_TEXT_RUN = 64
MAX_BRIDGE_GAP = 8
MIN_BINARY_CHUNK = 64     # binary chunks smaller than this get absorbed into text

# Binary blob compression methods
BLOB_GZIP = 0x47   # 'G'
BLOB_LZMA = 0x4C   # 'L'
BLOB_RAW = 0x52     # 'R'
LZMA_THRESHOLD = 4096  # use lzma above this size, gzip below

# Bytes considered "text-like": printable ASCII (32–126) + tab/LF/CR
TEXT_BYTES = frozenset(
    range(32, 127)
) | {9, 10, 13}


def _segment_chunks(data: bytes) -> list[tuple[int, int, int]]:
    """Segment data into text and binary chunks.

    Returns list of (chunk_type, offset, length) tuples where chunk_type
    is CHUNK_TYPE_TEXT or CHUNK_TYPE_BINARY.
    """
    if not data:
        return []

    # Step 1: classify each byte and collect contiguous runs
    runs = []  # list of (type, offset, length)
    current_type = CHUNK_TYPE_TEXT if data[0] in TEXT_BYTES else CHUNK_TYPE_BINARY
    run_start = 0

    for i in range(1, len(data)):
        byte_type = CHUNK_TYPE_TEXT if data[i] in TEXT_BYTES else CHUNK_TYPE_BINARY
        if byte_type != current_type:
            runs.append((current_type, run_start, i - run_start))
            current_type = byte_type
            run_start = i
    runs.append((current_type, run_start, len(data) - run_start))

    # Step 2: demote short text runs to binary
    runs = [
        (CHUNK_TYPE_BINARY if t == CHUNK_TYPE_TEXT and length < MIN_TEXT_RUN else t,
         off, length)
        for t, off, length in runs
    ]

    # Step 3: merge adjacent same-type runs (after demotion)
    merged = [runs[0]]
    for t, off, length in runs[1:]:
        if t == merged[-1][0]:
            prev_t, prev_off, prev_len = merged[-1]
            merged[-1] = (prev_t, prev_off, prev_len + length)
        else:
            merged.append((t, off, length))
    runs = merged

    # Step 4: bridge small binary gaps between text runs
    if len(runs) >= 3:
        bridged = [runs[0]]
        i = 1
        while i < len(runs) - 1:
            prev_t = bridged[-1][0]
            curr_t, curr_off, curr_len = runs[i]
            next_t = runs[i + 1][0]

            if (prev_t == CHUNK_TYPE_TEXT and curr_t == CHUNK_TYPE_BINARY
                    and next_t == CHUNK_TYPE_TEXT and curr_len <= MAX_BRIDGE_GAP):
                # Bridge: merge prev + gap + next into one text chunk
                prev_t2, prev_off, prev_len = bridged[-1]
                next_t2, next_off, next_len = runs[i + 1]
                bridged[-1] = (CHUNK_TYPE_TEXT, prev_off,
                               prev_len + curr_len + next_len)
                i += 2
            else:
                bridged.append((curr_t, curr_off, curr_len))
                i += 1
        if i < len(runs):
            bridged.append(runs[i])
        runs = bridged

    # Step 5: final merge of adjacent same-type runs
    merged = [runs[0]]
    for t, off, length in runs[1:]:
        if t == merged[-1][0]:
            prev_t, prev_off, prev_len = merged[-1]
            merged[-1] = (prev_t, prev_off, prev_len + length)
        else:
            merged.append((t, off, length))
    runs = merged

    # Step 6: absorb small binary chunks into adjacent text chunks
    if len(runs) >= 2:
        absorbed = []
        i = 0
        while i < len(runs):
            t, off, length = runs[i]
            if t == CHUNK_TYPE_BINARY and length < MIN_BINARY_CHUNK:
                left_text = (absorbed and absorbed[-1][0] == CHUNK_TYPE_TEXT)
                right_text = (i + 1 < len(runs)
                              and runs[i + 1][0] == CHUNK_TYPE_TEXT)
                if left_text and right_text:
                    # Merge left + this + right into one text chunk
                    prev_t, prev_off, prev_len = absorbed[-1]
                    _next_t, _next_off, next_len = runs[i + 1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length + next_len)
                    i += 2
                    continue
                elif left_text:
                    prev_t, prev_off, prev_len = absorbed[-1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length)
                    i += 1
                    continue
                elif right_text:
                    # Convert to text; will merge with next text chunk
                    absorbed.append((CHUNK_TYPE_TEXT, off, length))
                    i += 1
                    continue
            absorbed.append((t, off, length))
            i += 1
        runs = absorbed

        # Final merge after absorption
        merged = [runs[0]]
        for t, off, length in runs[1:]:
            if t == merged[-1][0]:
                prev_t, prev_off, prev_len = merged[-1]
                merged[-1] = (prev_t, prev_off, prev_len + length)
            else:
                merged.append((t, off, length))
        runs = merged

    return runs


class NeuralCompressor:
    """Lossless neural text compressor."""

    def __init__(self, model: ModelWrapper = None, verbose: bool = True):
        self.verbose = verbose
        self.model = model or ModelWrapper(verbose=verbose)

    def _compress_text_to_stream(
        self, text: str, *,
        bytes_done: int = 0, bytes_total: int = 0, chunk_size: int = 0,
    ) -> tuple[int, int, bytes]:
        """Arithmetic-code a text string using the model.

        Caller must call self.model.reset_cache() before this if needed.
        When bytes_total > 0, progress lines include overall completion %.

        Returns:
            (token_count, bit_count, stream_bytes)
        """
        token_ids = self.model.tokenizer.encode(text)
        num_tokens = len(token_ids)

        if self.verbose:
            print(f"Tokens: {num_tokens}", file=sys.stderr)

        encoder = ArithmeticEncoder()
        context = []

        for i, token_id in enumerate(token_ids):
            if self.verbose and (i % 100 == 0 or i == num_tokens - 1):
                line = (
                    f"\rCompressing: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)"
                )
                if bytes_total > 0:
                    frac = (i + 1) / num_tokens if num_tokens else 1
                    overall = (bytes_done + chunk_size * frac) / bytes_total
                    line += f"  [total: {100*overall:.1f}%]"
                print(line, end="", file=sys.stderr)

            probs = self.model.get_probs(context)
            cdf = probs_to_cdf(probs, CDF_TOTAL)
            encoder.encode_symbol(cdf, token_id)

            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK
                context = context[-keep:]

        if self.verbose:
            print(file=sys.stderr)

        compressed_bits = encoder.get_bit_count()
        stream = encoder.finish()
        return num_tokens, compressed_bits, stream

    def _decompress_text_stream(self, stream: bytes, num_tokens: int) -> str:
        """Decode an arithmetic-coded stream back to text.

        Caller must call self.model.reset_cache() before this if needed.

        Returns:
            Decoded text string.
        """
        decoder = ArithmeticDecoder(stream)
        context = []
        token_ids = []

        for i in range(num_tokens):
            if self.verbose and (i % 100 == 0 or i == num_tokens - 1):
                print(
                    f"\rDecompressing: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)",
                    end="", file=sys.stderr,
                )

            probs = self.model.get_probs(context)
            cdf = probs_to_cdf(probs, CDF_TOTAL)
            token_id = decoder.decode_symbol(cdf)
            token_ids.append(token_id)

            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK
                context = context[-keep:]

        if self.verbose:
            print(file=sys.stderr)

        return self.model.tokenizer.decode(token_ids)

    def compress(self, text: str) -> bytes:
        """Compress text to bytes.

        Args:
            text: Input text string.

        Returns:
            Compressed data as bytes (header + arithmetic-coded stream).
        """
        if not text:
            return MAGIC + struct.pack('>II', 0, 0)

        self.model.reset_cache()
        num_tokens, compressed_bits, stream = self._compress_text_to_stream(text)

        header = MAGIC + struct.pack('>II', num_tokens, compressed_bits)
        return header + stream

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes using hybrid chunked format (NC02v2).

        Text chunks are neural-compressed individually. All binary chunks
        are merged into one blob and compressed with gzip or lzma.

        Args:
            data: Raw binary data.

        Returns:
            Compressed data with NC02 header.
        """
        chunks = _segment_chunks(data)
        num_entries = len(chunks)

        # File header: magic + version + entry count
        file_header = MAGIC_BIN + struct.pack('>II', NC02_VERSION, num_entries)

        if num_entries == 0:
            return file_header

        # Build entry table and collect binary blob
        entry_table = []
        binary_parts = []
        text_indices = []  # indices into chunks that are text
        total_binary = 0

        for ci, (chunk_type, offset, length) in enumerate(chunks):
            entry_table.append(struct.pack('>BI', chunk_type, length))
            if chunk_type == CHUNK_TYPE_BINARY:
                binary_parts.append(data[offset:offset + length])
                total_binary += length
            else:
                text_indices.append(ci)

        # Compress merged binary blob
        if total_binary > 0:
            binary_blob = b''.join(binary_parts)

            if self.verbose:
                print(f"Binary blob: {total_binary} bytes", file=sys.stderr)

            if total_binary >= LZMA_THRESHOLD:
                compressed = lzma.compress(binary_blob)
                method = BLOB_LZMA
            else:
                compressed = gzip.compress(binary_blob, compresslevel=9)
                method = BLOB_GZIP

            if len(compressed) >= total_binary:
                compressed = binary_blob
                method = BLOB_RAW

            if self.verbose:
                labels = {BLOB_GZIP: "gzip", BLOB_LZMA: "lzma", BLOB_RAW: "raw"}
                print(f"Binary blob compressed: {len(compressed)} bytes "
                      f"({labels[method]})", file=sys.stderr)

            binary_section = struct.pack('>BI', method, len(compressed)) + compressed
        else:
            binary_section = b''

        # Compress text chunks (slow part — with progress tracking)
        total_bytes = len(data)
        text_bytes_done = total_binary  # binary is already handled
        text_streams = []

        for ti, ci in enumerate(text_indices):
            chunk_type, offset, length = chunks[ci]
            chunk_data = data[offset:offset + length]

            if self.verbose:
                overall = 100 * text_bytes_done / total_bytes if total_bytes else 0
                print(f"Text chunk {ti+1}/{len(text_indices)}, {length} bytes"
                      f"  [total: {overall:.1f}%]",
                      file=sys.stderr)

            text = chunk_data.decode('latin-1')
            self.model.reset_cache()
            token_count, bit_count, stream = self._compress_text_to_stream(
                text,
                bytes_done=text_bytes_done, bytes_total=total_bytes,
                chunk_size=length,
            )
            text_streams.append(
                struct.pack('>III', token_count, bit_count, len(stream)) + stream
            )
            text_bytes_done += length

        # Assemble: header + entry_table + binary_section + text_streams
        return (file_header
                + b''.join(entry_table)
                + binary_section
                + b''.join(text_streams))

    def decompress(self, data: bytes) -> "str | bytes":
        """Decompress bytes back to text or raw bytes.

        Args:
            data: Compressed data (as produced by compress() or compress_bytes()).

        Returns:
            Original text string (NC01) or raw bytes (NC02).
        """
        if len(data) < HEADER_SIZE:
            raise ValueError("Data too short to contain header")

        magic = data[:4]
        if magic not in (MAGIC, MAGIC_BIN):
            raise ValueError(f"Invalid magic bytes: {magic!r} (expected {MAGIC!r} or {MAGIC_BIN!r})")

        if magic == MAGIC:
            return self._decompress_nc01(data)
        return self._decompress_nc02(data)

    def _decompress_nc01(self, data: bytes) -> str:
        """Decompress NC01 (text) format."""
        num_tokens, _compressed_bits = struct.unpack('>II', data[4:HEADER_SIZE])

        if num_tokens == 0:
            return ""

        self.model.reset_cache()
        stream = data[HEADER_SIZE:]
        return self._decompress_text_stream(stream, num_tokens)

    def _decompress_nc02(self, data: bytes) -> bytes:
        """Decompress NC02v2 (hybrid chunked binary) format."""
        _version, num_entries = struct.unpack('>II', data[4:HEADER_SIZE])

        if num_entries == 0:
            return b""

        # Read entry table
        pos = HEADER_SIZE
        entries = []
        total_binary = 0
        for _ in range(num_entries):
            etype, elen = struct.unpack('>BI', data[pos:pos + 5])
            entries.append((etype, elen))
            if etype == CHUNK_TYPE_BINARY:
                total_binary += elen
            pos += 5

        # Read binary blob
        binary_data = b''
        if total_binary > 0:
            method, comp_len = struct.unpack('>BI', data[pos:pos + 5])
            pos += 5
            compressed = data[pos:pos + comp_len]
            pos += comp_len

            if method == BLOB_RAW:
                binary_data = compressed
            elif method == BLOB_GZIP:
                binary_data = gzip.decompress(compressed)
            elif method == BLOB_LZMA:
                binary_data = lzma.decompress(compressed)

            if self.verbose:
                labels = {BLOB_GZIP: "gzip", BLOB_LZMA: "lzma", BLOB_RAW: "raw"}
                print(f"Binary blob: {total_binary} bytes "
                      f"({labels.get(method, '?')})", file=sys.stderr)

        # Reconstruct output in original order
        binary_offset = 0
        output_parts = []

        for ci, (etype, elen) in enumerate(entries):
            if etype == CHUNK_TYPE_BINARY:
                output_parts.append(binary_data[binary_offset:binary_offset + elen])
                binary_offset += elen
            else:
                token_count, _bit_count, stream_len = struct.unpack(
                    '>III', data[pos:pos + 12])
                pos += 12
                stream = data[pos:pos + stream_len]
                pos += stream_len

                if self.verbose:
                    print(f"Text chunk, {elen} bytes", file=sys.stderr)

                self.model.reset_cache()
                text = self._decompress_text_stream(stream, token_count)
                output_parts.append(text.encode('latin-1'))

        return b''.join(output_parts)
